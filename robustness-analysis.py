# ============================================================
# Strict 5-Fold Robustness Evaluation
# PHQ-9 SBERT + Bi-LSTM
#
# Purpose:
#   Train fold-specific SBERT + Bi-LSTM models.
#   Apply robustness perturbations ONLY to each fold's held-out test set.
#   Report clean / incomplete / noisy / paraphrased performance.
#
# Conditions:
#   1. clean_baseline
#   2. missing_1_item
#   3. missing_3_items
#   4. light_typo_noise
#   5. moderate_typo_noise
#   6. rule_based_paraphrase
# ============================================================


# =========================
# Install for Colab
# =========================
!pip -q install -U sentence-transformers openpyxl tensorflow tf-keras

# =========================
# Imports
# =========================
import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    cohen_kappa_score,
    classification_report,
    accuracy_score
)
from scipy.stats import spearmanr

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# Config
# =========================
SEED = 42
K = 5

DATA_PATH = "/content/PHQ9_Student_Depression_Dataset_Aligned.xlsx"

QUESTION_COLS = [
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

SCORE_COL = "PHQ-9 Score"

# SBERT settings
MENTAL_MODEL_ID = "mental/mental-bert-base-uncased"
MAX_SEQ_LEN = 128
SBERT_BATCH_SIZE = 32
SBERT_EPOCHS = 2
SBERT_WARMUP_FRAC = 0.1
SBERT_OUT_BASE = "/content/robustness_cv_outputs/sbert_folds"

# Regressor settings
REG_BATCH_SIZE = 32
REG_EPOCHS = 30
REG_LR = 1e-4

# Robustness settings
N_ROBUST_REPEATS = 10
ROBUST_OUT_DIR = "/content/robustness_cv_outputs"
ROBUST_EXAMPLE_DIR = os.path.join(ROBUST_OUT_DIR, "perturbed_examples")

ROBUST_CONDITION_ORDER = [
    "clean_baseline",
    "missing_1_item",
    "missing_3_items",
    "light_typo_noise",
    "moderate_typo_noise",
    "rule_based_paraphrase",
]

os.makedirs(ROBUST_OUT_DIR, exist_ok=True)
os.makedirs(ROBUST_EXAMPLE_DIR, exist_ok=True)
os.makedirs(SBERT_OUT_BASE, exist_ok=True)


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# Load data
# =========================
df_students = pd.read_excel(DATA_PATH)

missing_cols = [c for c in QUESTION_COLS if c not in df_students.columns]
if missing_cols:
    raise ValueError(f"Missing question columns: {missing_cols}")

if SCORE_COL not in df_students.columns:
    raise ValueError(f"Could not find score column: {SCORE_COL}")

TEXT_COLS = QUESTION_COLS

# Normalized target: 0 to 1
y = (df_students[SCORE_COL].values / 27.0).astype(np.float32)


# =========================
# Severity helper
# =========================
def phq9_to_class(score_denorm):
    """
    Convert PHQ-9 score to severity class.
    Uses 1-5 labels to match the original training code.
    """
    if score_denorm <= 4:
        return 1      # Minimal
    elif score_denorm <= 9:
        return 2      # Mild
    elif score_denorm <= 14:
        return 3      # Moderate
    elif score_denorm <= 19:
        return 4      # Moderately severe
    else:
        return 5      # Severe


# Severity labels for stratified splitting
sev = np.array([phq9_to_class(s) for s in (y * 27.0)], dtype=np.int32)


# =========================
# General helpers
# =========================
def clean_text_cell(x):
    """Convert missing/null-like values to empty strings."""
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in ["nan", "none", "null"]:
        return ""
    return x


def build_pairs_from_students(
    df_students_sub,
    y_sub_norm,
    text_cols,
    n_pairs_per_q=300,
    seed=42,
    hard_frac=0.5
):
    """
    Build weakly supervised sentence pairs for SBERT fine-tuning.
    Pairing is done within the same PHQ-9 question column.
    Similarity label = 1 - absolute PHQ-9 score difference / 27.
    """
    rng = np.random.RandomState(seed)
    y_den = (y_sub_norm * 27.0).astype(np.float32)

    rows = []

    for q in text_cols:
        texts = df_students_sub[q].apply(clean_text_cell)
        idx_valid = np.where(texts.values != "")[0]

        if len(idx_valid) < 6:
            continue

        t = texts.values
        s = y_den

        sorted_idx = idx_valid[np.argsort(s[idx_valid])]

        n_hard = int(n_pairs_per_q * hard_frac)
        n_rand = n_pairs_per_q - n_hard

        # Random pairs
        for _ in range(n_rand):
            i, j = rng.choice(idx_valid, size=2, replace=False)
            diff = abs(float(s[i] - s[j]))
            label = 1.0 - (diff / 27.0)
            rows.append((q, t[i], t[j], float(np.clip(label, 0.0, 1.0))))

        # Hard negative-style pairs: low-score vs high-score
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

    pair_df = pd.DataFrame(rows, columns=["question", "text1", "text2", "label"])

    pair_df["text1"] = pair_df["text1"].astype(str).str.strip()
    pair_df["text2"] = pair_df["text2"].astype(str).str.strip()
    pair_df = pair_df[(pair_df["text1"] != "") & (pair_df["text2"] != "")]
    pair_df["label"] = pair_df["label"].astype(float).clip(0, 1)

    return pair_df


def make_sbert_student_split(df_train_students, sev_train, seed=42, dev_frac=0.2):
    """
    Student-level train/dev split inside each fold's training set for SBERT tuning.
    This avoids using held-out fold test students during SBERT fine-tuning.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=dev_frac,
        random_state=seed
    )

    idx = np.arange(len(df_train_students))
    tr_idx, dv_idx = next(sss.split(idx, sev_train))

    return tr_idx, dv_idx


def build_sbert():
    """
    Build MentalBERT inside a SentenceTransformer/SBERT structure.
    """
    word_embedding_model = models.Transformer(
        MENTAL_MODEL_ID,
        max_seq_length=MAX_SEQ_LEN
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device=device
    )


def to_input_examples(frame):
    return [
        InputExample(texts=[r.text1, r.text2], label=float(r.label))
        for r in frame.itertuples(index=False)
    ]


def make_early_stop():
    return EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )


def build_regressor(input_dim):
    """
    Bi-LSTM regressor.
    Input shape: 9 PHQ-9 item embeddings.
    """
    model = Sequential([
        Bidirectional(LSTM(128), input_shape=(9, input_dim)),
        Dense(1, activation=None)
    ])

    model.compile(
        optimizer=Adam(learning_rate=REG_LR),
        loss="mse",
        metrics=["mae"]
    )

    return model


def encode_students(sbert_model, df_sub):
    """
    Encode the 9 response columns into a tensor:
    shape = (n_students, 9, embedding_dim)
    """
    emb_list = []

    for q in TEXT_COLS:
        texts = df_sub[q].apply(clean_text_cell).tolist()

        embs = sbert_model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )

        emb_list.append(embs)

    return np.stack(emb_list, axis=1).astype(np.float32)


# =========================
# Robustness perturbation helpers
# =========================
def make_incomplete_input(input_df, rng, n_missing_items=1):
    """
    Randomly replace n PHQ-9 item responses per student with empty strings.
    """
    out = input_df.copy()

    for idx in out.index:
        cols_to_blank = rng.choice(TEXT_COLS, size=n_missing_items, replace=False)

        for col in cols_to_blank:
            out.at[idx, col] = ""

    return out


def mutate_word_typo(word, rng):
    """
    Apply one small typo operation:
    delete a character, swap adjacent characters, or duplicate a character.
    """
    if len(word) < 4:
        return word

    chars = list(word)
    op = rng.choice(["delete", "swap", "duplicate"])

    if op == "delete" and len(chars) > 3:
        pos = rng.integers(1, len(chars) - 1)
        del chars[pos]

    elif op == "swap" and len(chars) > 4:
        pos = rng.integers(1, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    elif op == "duplicate":
        pos = rng.integers(1, len(chars) - 1)
        chars.insert(pos, chars[pos])

    return "".join(chars)


def add_typo_noise_to_text(text, rng, word_noise_prob=0.10):
    """
    Add character-level typo noise to selected words in one response.
    """
    text = clean_text_cell(text)

    if not text:
        return text

    tokens = text.split()
    noisy_tokens = []

    for tok in tokens:
        if rng.random() < word_noise_prob and any(ch.isalpha() for ch in tok):
            noisy_tokens.append(mutate_word_typo(tok, rng))
        else:
            noisy_tokens.append(tok)

    return " ".join(noisy_tokens)


def make_noisy_input(input_df, rng, word_noise_prob=0.10):
    """
    Apply typo noise to every PHQ-9 response column.
    """
    out = input_df.copy()

    for col in TEXT_COLS:
        out[col] = out[col].apply(
            lambda x: add_typo_noise_to_text(
                x,
                rng,
                word_noise_prob=word_noise_prob
            )
        )

    return out


PARAPHRASE_MAP = {
    "sometimes": ["occasionally", "from time to time"],
    "often": ["frequently", "many times"],
    "usually": ["generally", "most of the time"],
    "rarely": ["seldom", "not often"],
    "always": ["constantly", "all the time"],
    "very": ["really", "quite"],
    "a little": ["slightly", "somewhat"],
    "a lot": ["greatly", "a great deal"],

    "feel": ["am feeling", "seem to feel"],
    "felt": ["was feeling"],
    "feeling": ["experiencing"],
    "things": ["activities"],
    "doing things": ["doing activities"],
    "interest": ["motivation"],
    "pleasure": ["enjoyment"],

    "down": ["low"],
    "depressed": ["sad"],
    "hopeless": ["without much hope"],

    "trouble": ["difficulty"],
    "falling asleep": ["getting to sleep"],
    "staying asleep": ["remaining asleep"],
    "sleep too much": ["sleep more than usual"],

    "tired": ["fatigued"],
    "little energy": ["low energy"],
    "poor appetite": ["low appetite"],
    "overeat": ["eat more than usual"],
    "overeating": ["eating more than usual"],

    "bad about myself": ["negative about myself"],
    "failure": ["unsuccessful"],

    "trouble concentrating": ["difficulty focusing"],
    "concentrating": ["focusing"],
    "reading": ["studying"],
    "work": ["tasks"],
    "watching television": ["watching TV"],

    "moving": ["acting"],
    "speaking": ["talking"],
    "slowly": ["more slowly"],
    "fidgety": ["restless"],
    "restless": ["fidgety"],
}


def preserve_case(original, replacement):
    """
    Roughly preserve capitalization after replacement.
    """
    if original.isupper():
        return replacement.upper()

    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]

    return replacement


def paraphrase_text_rule_based(text, rng, replace_prob=0.35, max_replacements=3):
    """
    Controlled lexical paraphrasing.
    This simulates wording shift without using an external LLM.
    """
    text = clean_text_cell(text)

    if not text:
        return text

    out = text
    replacements_done = 0

    # Replace longer phrases first.
    keys = sorted(PARAPHRASE_MAP.keys(), key=len, reverse=True)

    for key in keys:
        if replacements_done >= max_replacements:
            break

        pattern = re.compile(
            r"\b" + re.escape(key) + r"\b",
            flags=re.IGNORECASE
        )

        def repl(match):
            nonlocal replacements_done

            if replacements_done >= max_replacements:
                return match.group(0)

            if rng.random() > replace_prob:
                return match.group(0)

            replacement = rng.choice(PARAPHRASE_MAP[key])
            replacement = preserve_case(match.group(0), replacement)
            replacements_done += 1

            return replacement

        out = pattern.sub(repl, out)

    return out


def make_paraphrased_input(input_df, rng, replace_prob=0.35, max_replacements=3):
    """
    Apply rule-based paraphrasing to every PHQ-9 response column.
    """
    out = input_df.copy()

    for col in TEXT_COLS:
        out[col] = out[col].apply(
            lambda x: paraphrase_text_rule_based(
                x,
                rng,
                replace_prob=replace_prob,
                max_replacements=max_replacements
            )
        )

    return out


# =========================
# Evaluation helpers
# =========================
def evaluate_predictions_denorm(y_true_den, y_pred_den):
    """
    Compute regression and severity-classification metrics.
    """
    mae = mean_absolute_error(y_true_den, y_pred_den)
    rmse = np.sqrt(mean_squared_error(y_true_den, y_pred_den))

    rho, _ = spearmanr(y_true_den, y_pred_den)
    if np.isnan(rho):
        rho = 0.0

    true_cls = np.array([phq9_to_class(s) for s in y_true_den], dtype=np.int32)
    pred_cls = np.array([phq9_to_class(s) for s in y_pred_den], dtype=np.int32)

    qwk = cohen_kappa_score(true_cls, pred_cls, weights="quadratic")
    acc = accuracy_score(true_cls, pred_cls)

    report_dict = classification_report(
        true_cls,
        pred_cls,
        labels=[1, 2, 3, 4, 5],
        target_names=[
            "1-Minimal",
            "2-Mild",
            "3-Moderate",
            "4-Mod. Severe",
            "5-Severe"
        ],
        output_dict=True,
        zero_division=0
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Spearman": rho,
        "QWK": qwk,
        "Accuracy": acc,
        "Macro_Precision": report_dict["macro avg"]["precision"],
        "Macro_Recall": report_dict["macro avg"]["recall"],
        "Weighted_Precision": report_dict["weighted avg"]["precision"],
        "Weighted_Recall": report_dict["weighted avg"]["recall"],
        "Macro_F1": report_dict["macro avg"]["f1-score"],
        "Weighted_F1": report_dict["weighted avg"]["f1-score"],
    }


def run_robustness_for_fold(
    fold,
    sbert_model,
    reg_model,
    df_test_students,
    y_test_norm,
    original_test_idx
):
    """
    Evaluate clean and perturbed versions of the current fold's held-out test set.
    No retraining is done during robustness evaluation.
    """
    fold_rows = []
    fold_prediction_rows = []

    y_test_den = y_test_norm * 27.0

    robustness_conditions = [
        (
            "clean_baseline",
            lambda input_df, rng: input_df.copy(),
            1
        ),
        (
            "missing_1_item",
            lambda input_df, rng: make_incomplete_input(
                input_df,
                rng,
                n_missing_items=1
            ),
            N_ROBUST_REPEATS
        ),
        (
            "missing_3_items",
            lambda input_df, rng: make_incomplete_input(
                input_df,
                rng,
                n_missing_items=3
            ),
            N_ROBUST_REPEATS
        ),
        (
            "light_typo_noise",
            lambda input_df, rng: make_noisy_input(
                input_df,
                rng,
                word_noise_prob=0.10
            ),
            N_ROBUST_REPEATS
        ),
        (
            "moderate_typo_noise",
            lambda input_df, rng: make_noisy_input(
                input_df,
                rng,
                word_noise_prob=0.25
            ),
            N_ROBUST_REPEATS
        ),
        (
            "rule_based_paraphrase",
            lambda input_df, rng: make_paraphrased_input(
                input_df,
                rng,
                replace_prob=0.35,
                max_replacements=3
            ),
            N_ROBUST_REPEATS
        ),
    ]

    for condition_name, perturb_fn, n_repeats in robustness_conditions:
        for repeat in range(n_repeats):
            rng = np.random.default_rng(SEED + fold * 1000 + repeat)

            perturbed_df = perturb_fn(df_test_students.copy(), rng)

            # Save one perturbed example file per fold/condition.
            if repeat == 0:
                example_path = os.path.join(
                    ROBUST_EXAMPLE_DIR,
                    f"fold_{fold}_{condition_name}_examples.csv"
                )
                perturbed_df.to_csv(example_path, index=False)

            x_test_perturbed = encode_students(sbert_model, perturbed_df)

            y_pred_norm = reg_model.predict(
                x_test_perturbed,
                verbose=0
            ).flatten()

            y_pred_den = np.clip(y_pred_norm * 27.0, 0, 27)

            metrics = evaluate_predictions_denorm(y_test_den, y_pred_den)

            fold_rows.append({
                "Fold": fold,
                "Condition": condition_name,
                "Repeat": repeat,
                **metrics
            })

            for local_i, orig_i in enumerate(original_test_idx):
                fold_prediction_rows.append({
                    "Fold": fold,
                    "Condition": condition_name,
                    "Repeat": repeat,
                    "Original_Index": int(orig_i),
                    "True_Score": float(y_test_den[local_i]),
                    "Predicted_Score": float(y_pred_den[local_i]),
                    "True_Class": int(phq9_to_class(y_test_den[local_i])),
                    "Predicted_Class": int(phq9_to_class(y_pred_den[local_i])),
                })

            print(
                f"Robustness | Fold {fold} | {condition_name} "
                f"| Repeat {repeat + 1}/{n_repeats} "
                f"| MAE={metrics['MAE']:.3f} | RMSE={metrics['RMSE']:.3f} "
                f"| Weighted F1={metrics['Weighted_F1']:.3f}"
            )

    return fold_rows, fold_prediction_rows


# =========================
# Main strict CV robustness loop
# =========================
skf = StratifiedKFold(
    n_splits=K,
    shuffle=True,
    random_state=SEED
)

robustness_rows = []
robustness_prediction_rows = []

for fold, (train_idx, test_idx) in enumerate(
    skf.split(np.zeros(len(sev)), sev),
    start=1
):
    print(f"\n================ Fold {fold}/{K} ================\n")

    set_seed(SEED + fold)

    df_train_students = df_students.iloc[train_idx].reset_index(drop=True)
    df_test_students = df_students.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test = y[test_idx]

    sev_train = sev[train_idx]

    # -------------------------
    # 1. Build fold-specific SBERT train/dev pairs
    # -------------------------
    sbert_tr_idx, sbert_dv_idx = make_sbert_student_split(
        df_train_students,
        sev_train,
        seed=SEED + fold,
        dev_frac=0.2
    )

    df_sbert_tr = df_train_students.iloc[sbert_tr_idx].reset_index(drop=True)
    df_sbert_dv = df_train_students.iloc[sbert_dv_idx].reset_index(drop=True)

    y_sbert_tr = y_train[sbert_tr_idx]
    y_sbert_dv = y_train[sbert_dv_idx]

    sbert_train_df = build_pairs_from_students(
        df_sbert_tr,
        y_sbert_tr,
        text_cols=TEXT_COLS,
        n_pairs_per_q=300,
        seed=SEED + fold,
        hard_frac=0.5
    )

    sbert_dev_df = build_pairs_from_students(
        df_sbert_dv,
        y_sbert_dv,
        text_cols=TEXT_COLS,
        n_pairs_per_q=120,
        seed=SEED + fold + 999,
        hard_frac=0.5
    )

    print(
        f"Fold {fold}: SBERT pairs "
        f"train={len(sbert_train_df)}, dev={len(sbert_dev_df)}"
    )

    train_samples = to_input_examples(sbert_train_df)
    dev_samples = to_input_examples(sbert_dev_df)

    # -------------------------
    # 2. Fine-tune SBERT for current fold
    # -------------------------
    sbert_model = build_sbert()

    train_loader = DataLoader(
        train_samples,
        batch_size=SBERT_BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    train_loss = losses.CosineSimilarityLoss(sbert_model)

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples,
        name=f"fold{fold}-dev"
    )

    warmup_steps = int(
        len(train_loader) * SBERT_EPOCHS * SBERT_WARMUP_FRAC
    )

    fold_sbert_out = os.path.join(SBERT_OUT_BASE, f"fold_{fold}")

    sbert_model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=SBERT_EPOCHS,
        warmup_steps=warmup_steps,
        evaluator=dev_evaluator,
        evaluation_steps=max(50, len(train_loader) // 2),
        output_path=fold_sbert_out,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=torch.cuda.is_available()
    )

    # Reload best fold checkpoint
    sbert_model = SentenceTransformer(fold_sbert_out)

    # -------------------------
    # 3. Encode clean train/test students
    # -------------------------
    x_train = encode_students(sbert_model, df_train_students)
    x_test = encode_students(sbert_model, df_test_students)

    emb_dim = x_train.shape[-1]

    print(
        f"Fold {fold}: x_train={x_train.shape}, "
        f"x_test={x_test.shape}, emb_dim={emb_dim}"
    )

    # -------------------------
    # 4. Train Bi-LSTM regressor for current fold
    # -------------------------
    tf.keras.backend.clear_session()
    reg_model = build_regressor(input_dim=emb_dim)

    reg_model.fit(
        x_train,
        y_train,
        batch_size=REG_BATCH_SIZE,
        epochs=REG_EPOCHS,
        validation_split=0.2,
        callbacks=[make_early_stop()],
        verbose=0
    )

    # -------------------------
    # 5. Quick clean evaluation for sanity check
    # -------------------------
    y_pred_clean_norm = reg_model.predict(x_test, verbose=0).flatten()
    y_pred_clean_den = np.clip(y_pred_clean_norm * 27.0, 0, 27)
    y_test_den = y_test * 27.0

    clean_metrics = evaluate_predictions_denorm(
        y_test_den,
        y_pred_clean_den
    )

    print(
        f"\nClean test sanity check | Fold {fold} "
        f"| MAE={clean_metrics['MAE']:.3f} "
        f"| RMSE={clean_metrics['RMSE']:.3f} "
        f"| Spearman={clean_metrics['Spearman']:.4f} "
        f"| QWK={clean_metrics['QWK']:.4f} "
        f"| Weighted F1={clean_metrics['Weighted_F1']:.4f}\n"
    )

    # -------------------------
    # 6. Robustness evaluation on held-out test set
    # -------------------------
    fold_robust_rows, fold_robust_prediction_rows = run_robustness_for_fold(
        fold=fold,
        sbert_model=sbert_model,
        reg_model=reg_model,
        df_test_students=df_test_students,
        y_test_norm=y_test,
        original_test_idx=test_idx
    )

    robustness_rows.extend(fold_robust_rows)
    robustness_prediction_rows.extend(fold_robust_prediction_rows)


# =========================
# Save raw robustness results
# =========================
robustness_df = pd.DataFrame(robustness_rows)
robustness_predictions_df = pd.DataFrame(robustness_prediction_rows)

robustness_by_repeat_path = os.path.join(
    ROBUST_OUT_DIR,
    "robustness_metrics_by_fold_repeat.csv"
)

robustness_predictions_path = os.path.join(
    ROBUST_OUT_DIR,
    "robustness_predictions_by_sample.csv"
)

robustness_df.to_csv(robustness_by_repeat_path, index=False)
robustness_predictions_df.to_csv(robustness_predictions_path, index=False)

print(f"\nSaved robustness metrics by fold/repeat -> {robustness_by_repeat_path}")
print(f"Saved robustness predictions by sample -> {robustness_predictions_path}")


# =========================
# Fold-level summary
# =========================
robust_metric_cols = [
    "MAE",
    "RMSE",
    "Spearman",
    "QWK",
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Weighted_Precision",
    "Weighted_Recall",
    "Macro_F1",
    "Weighted_F1",
]

# Average repeats within each fold first.
# This gives each fold equal weight.
robustness_fold_level = (
    robustness_df
    .groupby(["Condition", "Fold"], as_index=False)[robust_metric_cols]
    .mean()
)

robustness_fold_level["Condition"] = pd.Categorical(
    robustness_fold_level["Condition"],
    categories=ROBUST_CONDITION_ORDER,
    ordered=True
)

robustness_fold_level = robustness_fold_level.sort_values(
    ["Condition", "Fold"]
)

robustness_fold_level_path = os.path.join(
    ROBUST_OUT_DIR,
    "robustness_metrics_fold_level.csv"
)

robustness_fold_level.to_csv(robustness_fold_level_path, index=False)

print(f"Saved fold-level robustness metrics -> {robustness_fold_level_path}")


# =========================
# Mean ± SD across folds
# =========================
robust_summary_rows = []

for condition, group in robustness_fold_level.groupby(
    "Condition",
    observed=True
):
    row = {"Condition": condition}

    for metric in robust_metric_cols:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)

    robust_summary_rows.append(row)

robust_summary_df = pd.DataFrame(robust_summary_rows)

robust_summary_df["Condition"] = pd.Categorical(
    robust_summary_df["Condition"],
    categories=ROBUST_CONDITION_ORDER,
    ordered=True
)

robust_summary_df = robust_summary_df.sort_values("Condition")

robust_summary_path = os.path.join(
    ROBUST_OUT_DIR,
    "robustness_summary_mean_std_across_folds.csv"
)

robust_summary_df.to_csv(robust_summary_path, index=False)

print(f"Saved robustness summary -> {robust_summary_path}")


# =========================
# Paper-ready table
# =========================
def mean_pm_std(mean, std, decimals=3):
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


paper_robust_table = pd.DataFrame({
    "Condition": robust_summary_df["Condition"].astype(str),

    "MAE": robust_summary_df.apply(
        lambda x: mean_pm_std(x["MAE_mean"], x["MAE_std"], 3),
        axis=1
    ),

    "RMSE": robust_summary_df.apply(
        lambda x: mean_pm_std(x["RMSE_mean"], x["RMSE_std"], 3),
        axis=1
    ),

    "Spearman": robust_summary_df.apply(
        lambda x: mean_pm_std(x["Spearman_mean"], x["Spearman_std"], 4),
        axis=1
    ),

    "QWK": robust_summary_df.apply(
        lambda x: mean_pm_std(x["QWK_mean"], x["QWK_std"], 4),
        axis=1
    ),

    "Weighted F1": robust_summary_df.apply(
        lambda x: mean_pm_std(x["Weighted_F1_mean"], x["Weighted_F1_std"], 4),
        axis=1
    ),
})

paper_robust_table_path = os.path.join(
    ROBUST_OUT_DIR,
    "robustness_table_for_paper.csv"
)

paper_robust_table.to_csv(paper_robust_table_path, index=False)

print(f"\nSaved paper-ready robustness table -> {paper_robust_table_path}")

print("\n--- Paper-ready Robustness Table ---")
display(paper_robust_table)


# =========================
# Console summary
# =========================
print("\n--- Robustness Summary: Mean ± SD across 5 held-out folds ---")

for _, row in robust_summary_df.iterrows():
    print(f"\nCondition: {row['Condition']}")
    print(f"MAE        : {row['MAE_mean']:.3f} ± {row['MAE_std']:.3f}")
    print(f"RMSE       : {row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}")
    print(f"Spearman   : {row['Spearman_mean']:.4f} ± {row['Spearman_std']:.4f}")
    print(f"QWK        : {row['QWK_mean']:.4f} ± {row['QWK_std']:.4f}")
    print(f"Accuracy   : {row['Accuracy_mean']:.4f} ± {row['Accuracy_std']:.4f}")
    print(f"Weighted F1: {row['Weighted_F1_mean']:.4f} ± {row['Weighted_F1_std']:.4f}")
