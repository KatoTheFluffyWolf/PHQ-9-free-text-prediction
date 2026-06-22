# Unified PHQ-9 ablation runner
# Change RUN_VARIANTS near the bottom to run the full model or any ablation variant.
#
# Variants included:
#   full                  = MentalBERT-SBERT fine-tuning + 9-item BiLSTM regressor
#   glove_mean_pool       = static GloVe sentence embeddings + 9-item BiLSTM regressor
#   domain_general_sbert  = domain-general BERT backbone + SBERT fine-tuning + 9-item BiLSTM regressor
#   no_sequence_head      = MentalBERT-SBERT fine-tuning + mean over 9 item embeddings + MLP regressor
#   no_finetuning         = frozen MentalBERT-SBERT encoder + 9-item BiLSTM regressor

import os
import re
import json
import random
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    cohen_kappa_score,
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from scipy.stats import spearmanr

import torch
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import matplotlib.pyplot as plt


# ============================================================
# 1. Global paths and experiment settings
# ============================================================

SEED = 42
K = 5

STUDENTS_XLSX = "/content/PHQ9_Student_Depression_Dataset_Aligned.xlsx"
OUTPUT_ROOT = "/content/phq9_ablation_outputs"

# For the GloVe ablation only.
# Download/extract the file first, then point this path to glove.6B.300d.txt.
GLOVE_PATH = "/content/glove.6B.300d.txt"

# If your Excel file uses different column names, manually replace TEXT_COLS
# with the 9 raw free-text response columns in PHQ item order.
TEXT_COLS = None

# Fallback used when TEXT_COLS is None and your columns match the original notebook.
PHQ_ITEM_COLUMNS = [
    "Do you have little interest or pleasure in doing things?",
    "Do you feel down, depressed, or hopeless?",
    "Do you have trouble falling or staying asleep, or do you sleep too much?",
    "Do you feel tired or have little energy?",
    "Do you have poor appetite or tend to overeat?",
    "Do you feel bad about yourself or that you are a failure or have let yourself or your family down?",
    "Do you have trouble concentrating on things, such as reading, work, or watching television?",
    "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?",
    "Have you had thoughts of self-harm or felt that you would be better off dead?",
]


# ============================================================
# 2. Ablation configuration
# ============================================================

EncoderType = Literal["sbert", "glove"]
RegressorType = Literal["bilstm", "mlp"]
PoolingType = Literal["sequence", "mean_questions"]


@dataclass
class AblationConfig:
    name: str
    encoder_type: EncoderType
    model_id: Optional[str] = None
    fine_tune_sbert: bool = False
    pooling: PoolingType = "sequence"
    regressor: RegressorType = "bilstm"

    max_seq_len: int = 128

    sbert_batch_size: int = 32
    sbert_epochs: int = 2
    sbert_warmup_frac: float = 0.1
    sbert_train_pairs_per_q: int = 300
    sbert_dev_pairs_per_q: int = 120
    sbert_hard_frac: float = 0.5
    sbert_dev_frac: float = 0.2

    reg_batch_size: int = 32
    reg_epochs: int = 30
    reg_lr: float = 1e-4
    early_stop_patience: int = 5

    encode_batch_size: int = 64
    save_embeddings: bool = False
    save_confusion_matrices: bool = True


VARIANTS: dict[str, AblationConfig] = {
    "full": AblationConfig(
        name="full",
        encoder_type="sbert",
        model_id="mental/mental-bert-base-uncased",
        fine_tune_sbert=True,
        pooling="sequence",
        regressor="bilstm",
    ),

    "glove_mean_pool": AblationConfig(
        name="glove_mean_pool",
        encoder_type="glove",
        fine_tune_sbert=False,
        pooling="sequence",
        regressor="bilstm",
    ),

    "domain_general_sbert": AblationConfig(
        name="domain_general_sbert",
        encoder_type="sbert",
        model_id="google-bert/bert-base-cased",
        fine_tune_sbert=True,
        pooling="sequence",
        regressor="bilstm",
    ),

    "no_sequence_head": AblationConfig(
        name="no_sequence_head",
        encoder_type="sbert",
        model_id="mental/mental-bert-base-uncased",
        fine_tune_sbert=True,
        pooling="mean_questions",
        regressor="mlp",
    ),

    "no_finetuning": AblationConfig(
        name="no_finetuning",
        encoder_type="sbert",
        model_id="mental/mental-bert-base-uncased",
        fine_tune_sbert=False,
        pooling="sequence",
        regressor="bilstm",
    ),
}


# ============================================================
# 3. Reproducibility and data loading
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phq9_to_class(score_denorm: float) -> int:
    if score_denorm <= 4:
        return 1
    if score_denorm <= 9:
        return 2
    if score_denorm <= 14:
        return 3
    if score_denorm <= 19:
        return 4
    return 5


def resolve_text_cols(df: pd.DataFrame) -> list[str]:
    if TEXT_COLS is not None:
        missing = [c for c in TEXT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"TEXT_COLS contains missing columns: {missing}")
        return list(TEXT_COLS)

    if all(q in df.columns for q in PHQ_ITEM_COLUMNS):
        return PHQ_ITEM_COLUMNS

    raise ValueError(
        "Could not infer the 9 free-text columns. Set TEXT_COLS manually near the top "
        "of the script, in the same order as the PHQ-9 items."
    )


def load_students(path: str) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df = pd.read_excel(path)
    text_cols = resolve_text_cols(df)

    if "PHQ-9 Score" not in df.columns:
        raise ValueError("Expected a target column named 'PHQ-9 Score'.")

    y = (df["PHQ-9 Score"].values / 27.0).astype(np.float32)
    sev = np.array([phq9_to_class(s) for s in y * 27.0], dtype=np.int32)
    return df, text_cols, y, sev


def clean_text_series(series: pd.Series) -> pd.Series:
    texts = series.fillna("").astype(str).str.strip()
    return texts.mask(texts.str.lower().isin(["nan", "none", "null"]), "")


# ============================================================
# 4. SBERT fine-tuning helpers
# ============================================================

def build_pairs_from_students(
    df_students_sub: pd.DataFrame,
    y_sub_norm: np.ndarray,
    text_cols: list[str],
    n_pairs_per_q: int = 300,
    seed: int = 42,
    hard_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Builds weakly labeled sentence pairs from a student subset.
    Pairing is done within each PHQ item column.
    Pair similarity = 1 - absolute difference in PHQ-9 total score / 27.
    """
    rng = np.random.RandomState(seed)
    y_den = (y_sub_norm * 27.0).astype(np.float32)

    rows = []
    for q in text_cols:
        texts = clean_text_series(df_students_sub[q])
        idx_valid = np.where(texts.values != "")[0]

        if len(idx_valid) < 6:
            continue

        t = texts.values
        s = y_den
        sorted_idx = idx_valid[np.argsort(s[idx_valid])]

        n_hard = int(n_pairs_per_q * hard_frac)
        n_rand = n_pairs_per_q - n_hard

        for _ in range(n_rand):
            i, j = rng.choice(idx_valid, size=2, replace=False)
            diff = abs(float(s[i] - s[j]))
            label = 1.0 - (diff / 27.0)
            rows.append((q, t[i], t[j], float(np.clip(label, 0.0, 1.0))))

        lo = sorted_idx[: max(2, len(sorted_idx) // 4)]
        hi = sorted_idx[-max(2, len(sorted_idx) // 4):]

        for _ in range(n_hard):
            i = int(rng.choice(lo))
            j = int(rng.choice(hi))
            if i == j:
                continue
            diff = abs(float(s[i] - s[j]))
            label = 1.0 - (diff / 27.0)
            rows.append((q, t[i], t[j], float(np.clip(label, 0.0, 1.0))))

    df = pd.DataFrame(rows, columns=["question", "text1", "text2", "label"])
    df["text1"] = df["text1"].astype(str).str.strip()
    df["text2"] = df["text2"].astype(str).str.strip()
    df = df[(df["text1"] != "") & (df["text2"] != "")]
    df["label"] = df["label"].astype(float).clip(0, 1)
    return df


def make_sbert_student_split(
    df_train_students: pd.DataFrame,
    sev_train: np.ndarray,
    seed: int = 42,
    dev_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed)
    idx = np.arange(len(df_train_students))
    tr_idx, dv_idx = next(sss.split(idx, sev_train))
    return tr_idx, dv_idx


def build_sbert(config: AblationConfig) -> SentenceTransformer:
    if config.model_id is None:
        raise ValueError("SBERT config requires model_id.")

    word_embedding_model = models.Transformer(
        config.model_id,
        max_seq_length=config.max_seq_len,
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)


def to_input_examples(frame: pd.DataFrame) -> list[InputExample]:
    return [
        InputExample(texts=[r.text1, r.text2], label=float(r.label))
        for r in frame.itertuples(index=False)
    ]


def fine_tune_sbert_for_fold(
    config: AblationConfig,
    df_train_students: pd.DataFrame,
    y_train: np.ndarray,
    sev_train: np.ndarray,
    text_cols: list[str],
    fold: int,
    output_dir: Path,
) -> SentenceTransformer:
    tr_idx, dv_idx = make_sbert_student_split(
        df_train_students,
        sev_train,
        seed=SEED + fold,
        dev_frac=config.sbert_dev_frac,
    )

    df_sbert_tr = df_train_students.iloc[tr_idx].reset_index(drop=True)
    df_sbert_dv = df_train_students.iloc[dv_idx].reset_index(drop=True)
    y_sbert_tr = y_train[tr_idx]
    y_sbert_dv = y_train[dv_idx]

    sbert_train_df = build_pairs_from_students(
        df_sbert_tr,
        y_sbert_tr,
        text_cols=text_cols,
        n_pairs_per_q=config.sbert_train_pairs_per_q,
        seed=SEED + fold,
        hard_frac=config.sbert_hard_frac,
    )
    sbert_dev_df = build_pairs_from_students(
        df_sbert_dv,
        y_sbert_dv,
        text_cols=text_cols,
        n_pairs_per_q=config.sbert_dev_pairs_per_q,
        seed=SEED + fold + 999,
        hard_frac=config.sbert_hard_frac,
    )

    if len(sbert_train_df) < 500 or len(sbert_dev_df) < 200:
        print(
            f"[Warn] Fold {fold}: few SBERT pairs "
            f"(train={len(sbert_train_df)}, dev={len(sbert_dev_df)})."
        )

    train_samples = to_input_examples(sbert_train_df)
    dev_samples = to_input_examples(sbert_dev_df)

    sbert_model = build_sbert(config)
    train_loader = DataLoader(
        train_samples,
        batch_size=config.sbert_batch_size,
        shuffle=True,
        drop_last=False,
    )

    train_loss = losses.CosineSimilarityLoss(sbert_model)
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples,
        name=f"{config.name}-fold{fold}-dev",
    )
    warmup_steps = int(
        len(train_loader) * config.sbert_epochs * config.sbert_warmup_frac
    )

    fold_out = output_dir / "sbert_folds" / f"fold_{fold}"
    fold_out.mkdir(parents=True, exist_ok=True)

    sbert_model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=config.sbert_epochs,
        warmup_steps=warmup_steps,
        evaluator=dev_evaluator,
        evaluation_steps=max(50, len(train_loader) // 2),
        output_path=str(fold_out),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(str(fold_out), device=device)


# ============================================================
# 5. Encoders
# ============================================================

_token_re = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    text = str(text).lower().strip()
    if not text or text == "nan":
        return []
    return _token_re.findall(text)


def infer_glove_dim(glove_path: str) -> int:
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) > 2:
                return len(parts) - 1
    raise ValueError("Could not infer GloVe dimension; file seems empty or malformed.")


def build_vocab_from_data(df_students: pd.DataFrame, text_cols: list[str]) -> set[str]:
    vocab = set()
    for col in text_cols:
        for text in df_students[col].astype(str).tolist():
            vocab.update(tokenize(text))
    return vocab


def load_glove_subset(glove_path: str, vocab: set[str], dim: int) -> dict[str, np.ndarray]:
    glove = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            if word in vocab:
                glove[word] = np.asarray(parts[1:], dtype=np.float32)
    return glove


def glove_sentence_embedding(
    text: str,
    glove_dict: dict[str, np.ndarray],
    dim: int,
) -> np.ndarray:
    vectors = [glove_dict[tok] for tok in tokenize(text) if tok in glove_dict]
    if not vectors:
        return np.zeros((dim,), dtype=np.float32)
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


def prepare_glove(df_students: pd.DataFrame, text_cols: list[str]) -> tuple[dict[str, np.ndarray], int]:
    if not os.path.exists(GLOVE_PATH):
        raise FileNotFoundError(
            f"GloVe file not found: {GLOVE_PATH}. "
            "Download/extract GloVe first or edit GLOVE_PATH."
        )

    glove_dim = infer_glove_dim(GLOVE_PATH)
    vocab = build_vocab_from_data(df_students, text_cols)
    glove = load_glove_subset(GLOVE_PATH, vocab, glove_dim)
    print(f"Loaded {len(glove):,}/{len(vocab):,} vocabulary words from GloVe (dim={glove_dim}).")
    return glove, glove_dim


def encode_students_sbert(
    sbert_model: SentenceTransformer,
    df_sub: pd.DataFrame,
    text_cols: list[str],
    config: AblationConfig,
) -> np.ndarray:
    # Shape before optional pooling: (n_samples, 9, emb_dim)
    arr = df_sub[text_cols].fillna("").astype(str).apply(lambda c: c.str.strip()).values
    arr = np.where(
        np.isin(np.char.lower(arr.astype(str)), ["nan", "none", "null"]),
        "",
        arr,
    )

    n_samples = arr.shape[0]
    flat_texts = arr.reshape(-1).tolist()

    embs = sbert_model.encode(
        flat_texts,
        batch_size=config.encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    x_seq = embs.reshape(n_samples, len(text_cols), -1).astype(np.float32)

    if config.pooling == "mean_questions":
        return x_seq.mean(axis=1).astype(np.float32)

    return x_seq


def encode_students_glove(
    df_sub: pd.DataFrame,
    text_cols: list[str],
    glove: dict[str, np.ndarray],
    glove_dim: int,
    config: AblationConfig,
) -> np.ndarray:
    emb_list = []
    for q in text_cols:
        q_texts = clean_text_series(df_sub[q]).tolist()
        embs = np.stack(
            [glove_sentence_embedding(text, glove, glove_dim) for text in q_texts],
            axis=0,
        )
        emb_list.append(embs)

    x_seq = np.stack(emb_list, axis=1).astype(np.float32)

    if config.pooling == "mean_questions":
        return x_seq.mean(axis=1).astype(np.float32)

    return x_seq


# ============================================================
# 6. Regressor heads and evaluation
# ============================================================

def make_early_stop(config: AblationConfig) -> EarlyStopping:
    return EarlyStopping(
        monitor="val_loss",
        patience=config.early_stop_patience,
        restore_best_weights=True,
    )


def build_regressor(input_shape: tuple[int, ...], config: AblationConfig) -> Sequential:
    if config.regressor == "bilstm":
        if len(input_shape) != 2:
            raise ValueError(
                f"BiLSTM expects input shape (9, dim), got {input_shape}. "
                "Use pooling='sequence' for BiLSTM."
            )

        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(128)),
            Dense(1, activation=None),
        ])

    elif config.regressor == "mlp":
        if len(input_shape) != 1:
            raise ValueError(
                f"MLP expects input shape (dim,), got {input_shape}. "
                "Use pooling='mean_questions' for MLP."
            )

        model = Sequential([
            Input(shape=input_shape),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1, activation=None),
        ])

    else:
        raise ValueError(f"Unknown regressor: {config.regressor}")

    model.compile(
        optimizer=Adam(learning_rate=config.reg_lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def evaluate_predictions(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_den = np.clip(y_pred * 27.0, 0, 27)
    y_test_den = y_test * 27.0

    mae = mean_absolute_error(y_test_den, y_pred_den)
    rmse = np.sqrt(mean_squared_error(y_test_den, y_pred_den))

    rho, _ = spearmanr(y_test_den, y_pred_den)
    if np.isnan(rho):
        rho = 0.0

    true_cls = np.array([phq9_to_class(s) for s in y_test_den], dtype=np.int32)
    pred_cls = np.array([phq9_to_class(s) for s in y_pred_den], dtype=np.int32)

    qwk = cohen_kappa_score(true_cls, pred_cls, weights="quadratic")
    acc = accuracy_score(true_cls, pred_cls)

    report_dict = classification_report(
        true_cls,
        pred_cls,
        labels=[1, 2, 3, 4, 5],
        target_names=["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": rho,
        "qwk": qwk,
        "accuracy": acc,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "true_cls": true_cls,
        "pred_cls": pred_cls,
    }


def save_confusion_matrix(true_cls: np.ndarray, pred_cls: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(true_cls, pred_cls, labels=[1, 2, 3, 4, 5])
    class_names = ["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_results(fold_rows: list[dict]) -> dict:
    metric_names = [
        "mae",
        "rmse",
        "spearman",
        "qwk",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    summary = {}
    for metric in metric_names:
        values = np.array([row[metric] for row in fold_rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return summary


# ============================================================
# 7. Cross-validation runner
# ============================================================

def run_cv(config: AblationConfig) -> tuple[pd.DataFrame, dict]:
    print(f"\n================ Running variant: {config.name} ================\n")
    print(json.dumps(asdict(config), indent=2))

    set_seed(SEED)
    df_students, text_cols, y, sev = load_students(STUDENTS_XLSX)

    variant_out = Path(OUTPUT_ROOT) / config.name
    variant_out.mkdir(parents=True, exist_ok=True)

    glove = None
    glove_dim = None
    if config.encoder_type == "glove":
        glove, glove_dim = prepare_glove(df_students, text_cols)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(sev)), sev), start=1):
        print(f"\n---------------- Fold {fold}/{K}: {config.name} ----------------")
        set_seed(SEED + fold)

        df_train_students = df_students.iloc[train_idx].reset_index(drop=True)
        df_test_students = df_students.iloc[test_idx].reset_index(drop=True)

        y_train = y[train_idx]
        y_test = y[test_idx]
        sev_train = sev[train_idx]

        # ----- Encoder -----
        if config.encoder_type == "sbert":
            if config.fine_tune_sbert:
                encoder = fine_tune_sbert_for_fold(
                    config=config,
                    df_train_students=df_train_students,
                    y_train=y_train,
                    sev_train=sev_train,
                    text_cols=text_cols,
                    fold=fold,
                    output_dir=variant_out,
                )
            else:
                encoder = build_sbert(config)
                encoder.eval()

            x_train = encode_students_sbert(encoder, df_train_students, text_cols, config)
            x_test = encode_students_sbert(encoder, df_test_students, text_cols, config)

        elif config.encoder_type == "glove":
            if glove is None or glove_dim is None:
                raise RuntimeError("GloVe was not prepared.")
            x_train = encode_students_glove(df_train_students, text_cols, glove, glove_dim, config)
            x_test = encode_students_glove(df_test_students, text_cols, glove, glove_dim, config)

        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

        print(f"Fold {fold}: x_train={x_train.shape}, x_test={x_test.shape}")

        if config.save_embeddings:
            emb_path = variant_out / "embeddings" / f"fold_{fold}_embeddings.pkl"
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(emb_path, "wb") as f:
                pickle.dump(
                    {
                        "variant": config.name,
                        "fold": fold,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "x_train": x_train,
                        "x_test": x_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "input_shape": x_train.shape[1:],
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        # ----- Regressor -----
        tf.keras.backend.clear_session()
        reg_model = build_regressor(input_shape=x_train.shape[1:], config=config)

        reg_model.fit(
            x_train,
            y_train,
            batch_size=config.reg_batch_size,
            epochs=config.reg_epochs,
            validation_split=0.2,
            callbacks=[make_early_stop(config)],
            verbose=0,
        )

        reg_path = variant_out / "regressors" / f"regressor_fold_{fold}.keras"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_model.save(reg_path)

        # ----- Evaluation -----
        y_pred = reg_model.predict(x_test, verbose=0).flatten()
        metrics = evaluate_predictions(y_test, y_pred)

        if config.save_confusion_matrices:
            save_confusion_matrix(
                metrics["true_cls"],
                metrics["pred_cls"],
                variant_out / "confusion_matrices" / f"fold_{fold}.png",
                title=f"{config.name} - Fold {fold}/{K}",
            )

        row = {
            "variant": config.name,
            "fold": fold,
            **{k: v for k, v in metrics.items() if k not in {"true_cls", "pred_cls"}},
        }
        fold_rows.append(row)

        print(
            f"Fold {fold}/{K} -> "
            f"MAE: {row['mae']:.2f} | "
            f"RMSE: {row['rmse']:.2f} | "
            f"Spearman: {row['spearman']:.4f} | "
            f"QWK: {row['qwk']:.4f} | "
            f"Acc: {row['accuracy']:.4f}"
        )

        print(
            classification_report(
                metrics["true_cls"],
                metrics["pred_cls"],
                labels=[1, 2, 3, 4, 5],
                target_names=[
                    "1-Minimal",
                    "2-Mild",
                    "3-Moderate",
                    "4-Mod. Severe",
                    "5-Severe",
                ],
                digits=4,
                zero_division=0,
            )
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = summarize_results(fold_rows)
    summary["variant"] = config.name

    fold_df.to_csv(variant_out / "fold_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(variant_out / "summary_metrics.csv", index=False)

    with open(variant_out / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n--- Summary: {config.name} ---")
    print(f"MAE       : {summary['mae_mean']:.2f} ± {summary['mae_std']:.2f}")
    print(f"RMSE      : {summary['rmse_mean']:.2f} ± {summary['rmse_std']:.2f}")
    print(f"Spearman  : {summary['spearman_mean']:.4f} ± {summary['spearman_std']:.4f}")
    print(f"QWK       : {summary['qwk_mean']:.4f} ± {summary['qwk_std']:.4f}")
    print(f"Accuracy  : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Macro F1  : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1_mean']:.4f} ± {summary['weighted_f1_std']:.4f}")
    print(f"Saved outputs to: {variant_out}")

    return fold_df, summary


# ============================================================
# 8. Run selected variants
# ============================================================

# Run one variant:
# RUN_VARIANTS = ["full"]
# RUN_VARIANTS = ["glove_mean_pool"]
# RUN_VARIANTS = ["domain_general_sbert"]
# RUN_VARIANTS = ["no_sequence_head"]
# RUN_VARIANTS = ["no_finetuning"]

# Run all paper variants. This is expensive because several variants fine-tune SBERT.
RUN_VARIANTS = ["full"]

all_fold_dfs = []
all_summaries = []

for variant_name in RUN_VARIANTS:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {list(VARIANTS)}")

    fold_df, summary = run_cv(VARIANTS[variant_name])
    all_fold_dfs.append(fold_df)
    all_summaries.append(summary)

if all_fold_dfs:
    all_folds = pd.concat(all_fold_dfs, ignore_index=True)
    all_summary = pd.DataFrame(all_summaries)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(Path(OUTPUT_ROOT) / "all_fold_metrics.csv", index=False)
    all_summary.to_csv(Path(OUTPUT_ROOT) / "all_summary_metrics.csv", index=False)

    print("\n================ All selected variants summary ================")
    display_cols = [
        "variant",
        "mae_mean", "mae_std",
        "rmse_mean", "rmse_std",
        "spearman_mean", "spearman_std",
        "qwk_mean", "qwk_std",
        "weighted_f1_mean", "weighted_f1_std",
    ]
    print(all_summary[display_cols].to_string(index=False))
