# =========================
# Install (Colab)
# =========================
!pip -q install -U sentence-transformers openpyxl tensorflow tf-keras
from google.colab import files

import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, classification_report, accuracy_score, confusion_matrix
from scipy.stats import spearmanr

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import seaborn as sns
import matplotlib.pyplot as plt

import pickle

# =========================
# Config
# =========================
SEED = 42
K = 5

STUDENTS_XLSX = "/content/PHQ9_Student_Depression_Dataset_Aligned.xlsx"

MENTAL_MODEL_ID = "mental/mental-bert-base-uncased"
MAX_SEQ_LEN = 128
SBERT_BATCH_SIZE = 32
SBERT_EPOCHS = 2
SBERT_WARMUP_FRAC = 0.1
SBERT_OUT_BASE = "/content/sbert_folds"  # each fold saved separately

REG_BATCH_SIZE = 32
REG_EPOCHS = 30
REG_LR = 1e-4
REG_OUT_DIR = "/content/regressor_folds"

HISTORY_OUT_DIR = "/content/learning_curves"
os.makedirs(HISTORY_OUT_DIR, exist_ok=True)
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
df_students = pd.read_excel(STUDENTS_XLSX) #Dataset


# Questions list (also used to locate text columns in df_students)
Questions = [
    "Do you have little interest or pleasure in doing things?",
    "Do you feel down, depressed, or hopeless?",
    "Do you have trouble falling or staying asleep, or do you sleep too much?",
    "Do you feel tired or have little energy?",
    "Do you have poor appetite or tend to overeat?",
    "Do you feel bad about yourself or that you are a failure or have let yourself or your family down?",
    "Do you have trouble concentrating on things, such as reading, work, or watching television?",
    "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?",
    "Have you had thoughts of self-harm or felt that you would be better off dead?"
]

# -------------------------
# IMPORTANT: locate the 9 raw-text columns in df_students
# -------------------------
# Option A: your Excel columns match the question strings exactly:
if all(q in df_students.columns for q in Questions):
    TEXT_COLS = Questions
else:
    # Option B: edit this mapping to match your Excel (example placeholders)
    # e.g. TEXT_COLS = ["Q1_text","Q2_text",...]
    raise ValueError(
        "Could not find raw-text columns for all 9 questions in df_students.\n"
        "Either name your columns exactly as the Questions list, or define TEXT_COLS mapping manually."
    )

# Target y (normalized 0..1)
y = (df_students["PHQ-9 Score"].values / 27.0).astype(np.float32)

# Severity bins for stratification
def phq9_to_class(score_denorm):
    if score_denorm <= 4: return 1
    elif score_denorm <= 9: return 2
    elif score_denorm <= 14: return 3
    elif score_denorm <= 19: return 4
    else: return 5

sev = np.array([phq9_to_class(s) for s in (y * 27)], dtype=np.int32)

# =========================
# Helpers
# =========================

def build_pairs_from_students(
    df_students_sub: pd.DataFrame,
    y_sub_norm: np.ndarray,              # normalized 0..1, same length as df_students_sub
    text_cols: list[str],
    n_pairs_per_q: int = 300,            # ~300 * 9 = 2700 pairs per SBERT split
    seed: int = 42,
    hard_frac: float = 0.5               # fraction of "hard negatives" (far score pairs)
) -> pd.DataFrame:
    """
    Builds weakly-labeled sentence pairs from a student subset.
    Pairing is done per question column (so text1/text2 are answers to the same PHQ item).
    Similarity label is based on PHQ-9 total score closeness: 1 - |diff|/27.

    Returns DataFrame columns: question, text1, text2, label
    """
    rng = np.random.RandomState(seed)
    y_den = (y_sub_norm * 27.0).astype(np.float32)

    rows = []
    for q in text_cols:
        texts = df_students_sub[q].fillna("").astype(str).str.strip()
        texts = texts.mask(texts.str.lower().isin(["nan", "none", "null"]), "")
        idx_valid = np.where(texts.values != "")[0]

        if len(idx_valid) < 6:
            continue

        # Use arrays for fast indexing
        t = texts.values
        s = y_den

        # Pre-sort indices by score for easy "near" vs "far" sampling
        sorted_idx = idx_valid[np.argsort(s[idx_valid])]

        n_hard = int(n_pairs_per_q * hard_frac)
        n_rand = n_pairs_per_q - n_hard

        # ---- Random pairs ----
        for _ in range(n_rand):
            i, j = rng.choice(idx_valid, size=2, replace=False)
            diff = abs(float(s[i] - s[j]))
            label = 1.0 - (diff / 27.0)
            rows.append((q, t[i], t[j], float(np.clip(label, 0.0, 1.0))))

        # ---- "Hard" pairs: force large score differences (more contrast) ----
        # Pair from low-score end with high-score end
        lo = sorted_idx[: max(2, len(sorted_idx)//4)]
        hi = sorted_idx[-max(2, len(sorted_idx)//4) :]

        for _ in range(n_hard):
            i = int(rng.choice(lo))
            j = int(rng.choice(hi))
            if i == j:
                continue
            diff = abs(float(s[i] - s[j]))
            label = 1.0 - (diff / 27.0)  # will be small for far pairs
            rows.append((q, t[i], t[j], float(np.clip(label, 0.0, 1.0))))

    df = pd.DataFrame(rows, columns=["question", "text1", "text2", "label"])
    # final cleanup
    df["text1"] = df["text1"].astype(str).str.strip()
    df["text2"] = df["text2"].astype(str).str.strip()
    df = df[(df["text1"] != "") & (df["text2"] != "")]
    df["label"] = df["label"].astype(float).clip(0, 1)
    return df

def make_sbert_student_split(df_train_students, sev_train, seed=42, dev_frac=0.2):
    """
    Student-level split inside the fold's training students for SBERT tuning.
    This avoids overlapping students/texts between SBERT-train and SBERT-dev.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed)
    idx = np.arange(len(df_train_students))
    tr_idx, dv_idx = next(sss.split(idx, sev_train))
    return tr_idx, dv_idx

def build_sbert():
    word_embedding_model = models.Transformer(MENTAL_MODEL_ID, max_seq_length=MAX_SEQ_LEN)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

def to_input_examples(frame):
    return [InputExample(texts=[r.text1, r.text2], label=float(r.label))
            for r in frame.itertuples(index=False)]

def make_early_stop():
    return EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

def build_regressor(input_dim):
    # (9, dim) input expected
    model = Sequential([
        Bidirectional(LSTM(128), input_shape=(9, input_dim)),
        Dense(1, activation=None)
    ])
    model.compile(optimizer=Adam(learning_rate=REG_LR), loss="mse", metrics=["mae"])
    return model

def encode_students(sbert_model, df_sub):
    emb_list = []
    for q in TEXT_COLS:
        texts = df_sub[q].fillna("").astype(str).str.strip()
        texts = texts.mask(texts.str.lower().isin(["nan", "none", "null"]), "")
        q_texts = texts.tolist()

        embs = sbert_model.encode(
            q_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )
        emb_list.append(embs)

    return np.stack(emb_list, axis=1).astype(np.float32)

def history_to_df(history, fold):
    """
    Convert Keras History object to a DataFrame.
    Loss is MSE on normalized PHQ-9 score.
    MAE is normalized MAE; mae_phq9 and val_mae_phq9 convert it back to PHQ-9 points.
    """
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.insert(0, "fold", fold)

    if "mae" in hist.columns:
        hist["mae_phq9"] = hist["mae"] * 27.0
    if "val_mae" in hist.columns:
        hist["val_mae_phq9"] = hist["val_mae"] * 27.0

    return hist


def plot_fold_learning_curve(hist_df, fold, out_dir):
    """
    Plot training vs validation MAE for one fold.
    Uses PHQ-9 points instead of normalized MAE because it is easier to explain in the paper.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(hist_df["epoch"], hist_df["mae_phq9"], marker="o", label="Training MAE")
    plt.plot(hist_df["epoch"], hist_df["val_mae_phq9"], marker="o", label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE in PHQ-9 points")
    plt.title(f"Learning Curve of Bi-LSTM Regressor (Fold {fold})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"learning_curve_fold_{fold}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curve -> {out_path}")


def plot_average_learning_curve(all_histories_df, out_dir):
    """
    Plot average training/validation MAE across folds.
    Handles early stopping by averaging only available folds at each epoch.
    """
    avg = (
        all_histories_df
        .groupby("epoch")[["mae_phq9", "val_mae_phq9"]]
        .agg(["mean", "std", "count"])
    )

    epochs = avg.index.values

    train_mean = avg[("mae_phq9", "mean")].values
    train_std = avg[("mae_phq9", "std")].fillna(0).values

    val_mean = avg[("val_mae_phq9", "mean")].values
    val_std = avg[("val_mae_phq9", "std")].fillna(0).values

    plt.figure(figsize=(7, 5))

    plt.plot(epochs, train_mean, marker="o", label="Training MAE")
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15)

    plt.plot(epochs, val_mean, marker="o", label="Validation MAE")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("MAE in PHQ-9 points")
    plt.title("Average Learning Curve Across 5 Folds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "learning_curve_average_5fold.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved average learning curve -> {out_path}")

# =========================
# K-Fold CV: SBERT fine-tune -> embed -> regressor train/eval
# =========================
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

fold_mae, fold_rmse, fold_spearman, fold_qwk = [], [], [], []
fold_acc, fold_macro_f1, fold_weighted_f1, fold_macro_recall, fold_macro_precision, fold_weighted_recall, fold_weighted_precision = [], [], [], [], [], [], []

# Create output directories

os.makedirs(SBERT_OUT_BASE, exist_ok=True)
CM_OUT_DIR = "/content/confusion_matrices"
os.makedirs(CM_OUT_DIR, exist_ok=True)

all_history_dfs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(sev)), sev), start=1):
    print(f"\n================ Fold {fold}/{K} ================ রাহুল")
    set_seed(SEED + fold)

    df_train_students = df_students.iloc[train_idx].reset_index(drop=True)
    df_test_students  = df_students.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test  = y[test_idx]

   # ---------
    # 1) Build fold-specific SBERT pairs directly from fold TRAIN students
    #    (student-level split inside the training fold -> SBERT train/dev)
    # ---------
    sev_train = sev[train_idx]  # strat labels for students in this fold's train split
    sbert_tr_idx, sbert_dv_idx = make_sbert_student_split(
        df_train_students, sev_train, seed=SEED + fold, dev_frac=0.2
    )

    df_sbert_tr = df_train_students.iloc[sbert_tr_idx].reset_index(drop=True)
    df_sbert_dv = df_train_students.iloc[sbert_dv_idx].reset_index(drop=True)

    y_sbert_tr = y_train[sbert_tr_idx]
    y_sbert_dv = y_train[sbert_dv_idx]

    # Build pairs (tune these counts based on speed)
    sbert_train_df = build_pairs_from_students(
        df_sbert_tr, y_sbert_tr,
        text_cols=TEXT_COLS,
        n_pairs_per_q=300,
        seed=SEED + fold,
        hard_frac=0.5
    )
    sbert_dev_df = build_pairs_from_students(
        df_sbert_dv, y_sbert_dv,
        text_cols=TEXT_COLS,
        n_pairs_per_q=120,
        seed=SEED + fold + 999,
        hard_frac=0.5
    )

    if len(sbert_train_df) < 500 or len(sbert_dev_df) < 200:
        print(f"[Warn] Fold {fold}: few SBERT pairs (train={len(sbert_train_df)}, dev={len(sbert_dev_df)}). "
              f"Consider increasing n_pairs_per_q or check missing texts.")

    train_samples = to_input_examples(sbert_train_df)
    dev_samples   = to_input_examples(sbert_dev_df)

    # ---------
    # 2) Build & fine-tune SBERT for this fold
    # ---------
    sbert_model = build_sbert()
    train_loader = DataLoader(train_samples, batch_size=SBERT_BATCH_SIZE, shuffle=True, drop_last=False)
    train_loss = losses.CosineSimilarityLoss(sbert_model)

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name=f"fold{fold}-dev")
    warmup_steps = int(len(train_loader) * SBERT_EPOCHS * SBERT_WARMUP_FRAC)

    fold_out = os.path.join(SBERT_OUT_BASE, f"fold_{fold}")
    sbert_model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=SBERT_EPOCHS,
        warmup_steps=warmup_steps,
        evaluator=dev_evaluator,
        evaluation_steps=max(50, len(train_loader)//2),
        output_path=fold_out,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=torch.cuda.is_available()
    )
    sbert_model = SentenceTransformer(fold_out)  # reload best checkpoint

    # ---------
    # 3) Encode train/test students with fold-specific SBERT
    # ---------
    x_train = encode_students(sbert_model, df_train_students)
    x_test  = encode_students(sbert_model, df_test_students)

    #Save the embeddings for each fold
    emb_path = os.path.join(SBERT_OUT_BASE, f"fold_{fold}_embeddings.pkl")
    with open(emb_path, "wb") as f:
        pickle.dump(
            {
                "fold": fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
                "emb_dim": int(x_train.shape[-1]),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"Saved embeddings -> {emb_path}")

    emb_dim = x_train.shape[-1]
    print(f"Fold {fold}: embeddings shape train={x_train.shape}, test={x_test.shape} (dim={emb_dim})")

    # ---------
    # 4) Train regressor head (fresh per fold)
    # ---------
    tf.keras.backend.clear_session()
    reg_model = build_regressor(input_dim=emb_dim)

    history = reg_model.fit(
        x_train, y_train,
        batch_size=REG_BATCH_SIZE,
        epochs=REG_EPOCHS,
        validation_split=0.2,
        callbacks=[make_early_stop()],
        verbose=0
    )

    hist_df = history_to_df(history, fold)
    all_history_dfs.append(hist_df)

    hist_csv_path = os.path.join(HISTORY_OUT_DIR, f"history_fold_{fold}.csv")
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"Saved training history -> {hist_csv_path}")

    plot_fold_learning_curve(hist_df, fold, HISTORY_OUT_DIR)

    os.makedirs(REG_OUT_DIR, exist_ok=True)

    reg_path = os.path.join(REG_OUT_DIR, f"regressor_fold_{fold}.h5")
    reg_model.save(reg_path)
    print(f"Saved regressor head -> {reg_path}")


    # ---------
    # 5) Evaluate
    # ---------
    y_pred = reg_model.predict(x_test, verbose=0).flatten()

    y_pred_den = y_pred * 27
    y_pred_den = np.clip(y_pred_den, 0, 27)
    y_test_den = y_test * 27

    mae = mean_absolute_error(y_test_den, y_pred_den)
    rmse = np.sqrt(mean_squared_error(y_test_den, y_pred_den))

    rho, _ = spearmanr(y_test_den, y_pred_den)
    if np.isnan(rho): rho = 0.0

    true_cls = np.array([phq9_to_class(s) for s in y_test_den], dtype=np.int32)
    pred_cls = np.array([phq9_to_class(s) for s in y_pred_den], dtype=np.int32)

    qwk = cohen_kappa_score(true_cls, pred_cls, weights="quadratic")

    acc = accuracy_score(true_cls, pred_cls)
    report_dict = classification_report(
        true_cls, pred_cls,
        labels=[1,2,3,4,5],
        target_names=["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"],
        output_dict=True,
        zero_division=0
    )
    macro_precision   = report_dict["macro avg"]["precision"]
    macro_recall      = report_dict["macro avg"]["recall"]
    weighted_precision = report_dict["weighted avg"]["precision"]
    weighted_recall    = report_dict["weighted avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    cm = confusion_matrix(true_cls, pred_cls, labels=[1,2,3,4,5])

    class_names = ["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"]

    cm_path = os.path.join(CM_OUT_DIR, f"confusion_matrix_fold_{fold}.png")

    plt.figure(figsize=(7,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Confusion Matrix (Fold {fold}/{K})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved CM -> {cm_path}")


    fold_mae.append(mae)
    fold_rmse.append(rmse)
    fold_spearman.append(rho)
    fold_qwk.append(qwk)
    fold_acc.append(acc)
    fold_macro_f1.append(macro_f1)
    fold_weighted_f1.append(weighted_f1)
    fold_macro_precision.append(macro_precision)
    fold_macro_recall.append(macro_recall)
    fold_weighted_precision.append(weighted_precision)
    fold_weighted_recall.append(weighted_recall)

    print(f"Fold {fold}/{K} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | Spearman: {rho:.4f} | QWK: {qwk:.4f} | Acc: {acc:.4f}")
    print("\nClassification Report (Fold {}) ".format(fold))
    print(classification_report(
        true_cls, pred_cls,
        labels=[1,2,3,4,5],
        target_names=["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"],
        digits=4,
        zero_division=0
    ))
    print("Confusion Matrix (Fold {}) ".format(fold))
    print(cm)
    print("-" * 60)

print("\n--- Stratified K-Fold Summary (End-to-End) ---")
print(f"MAE     : {np.mean(fold_mae):.2f} \u00b1 {np.std(fold_mae, ddof=1):.2f}")
print(f"RMSE    : {np.mean(fold_rmse):.2f} \u00b1 {np.std(fold_rmse, ddof=1):.2f}")
print(f"Spearman: {np.mean(fold_spearman):.4f} \u00b1 {np.std(fold_spearman, ddof=1):.4f}")
print(f"QWK     : {np.mean(fold_qwk):.4f} \u00b1 {np.std(fold_qwk, ddof=1):.4f}")

print("\n--- Severity Classification Summary ---")
print(f"Accuracy     : {np.mean(fold_acc):.4f} \u00b1 {np.std(fold_acc, ddof=1):.4f}")
print(f"Macro Precision : {np.mean(fold_macro_precision):.4f} \u00b1 {np.std(fold_macro_precision, ddof=1):.4f}")
print(f"Macro Recall    : {np.mean(fold_macro_recall):.4f} \u00b1 {np.std(fold_macro_recall, ddof=1):.4f}")
print(f"Weighted Precision: {np.mean(fold_weighted_precision):.4f} \u00b1 {np.std(fold_weighted_precision, ddof=1):.4f}")
print(f"Weighted Recall   : {np.mean(fold_weighted_recall):.4f} \u00b1 {np.std(fold_weighted_recall, ddof=1):.4f}")
print(f"Macro F1     : {np.mean(fold_macro_f1):.4f} \u00b1 {np.std(fold_macro_f1, ddof=1):.4f}")
print(f"Weighted F1  : {np.mean(fold_weighted_f1):.4f} \u00b1 {np.std(fold_weighted_f1, ddof=1):.4f}")

all_histories_df = pd.concat(all_history_dfs, ignore_index=True)

all_histories_csv = os.path.join(HISTORY_OUT_DIR, "history_all_folds.csv")
all_histories_df.to_csv(all_histories_csv, index=False)
print(f"Saved all fold histories -> {all_histories_csv}")

plot_average_learning_curve(all_histories_df, HISTORY_OUT_DIR)

# =========================
# Final training on ALL data (after K-fold)
# =========================
print("\n================ Final Training on ALL data =================\n")
set_seed(SEED + 999)

# ---------
# 1) Build SBERT pairs using ALL students (with internal student split)
# ---------
sev_all = sev
sbert_tr_idx_all, sbert_dv_idx_all = make_sbert_student_split(
    df_students, sev_all, seed=SEED + 999, dev_frac=0.2
)

df_sbert_tr_all = df_students.iloc[sbert_tr_idx_all].reset_index(drop=True)
df_sbert_dv_all = df_students.iloc[sbert_dv_idx_all].reset_index(drop=True)

y_tr_all = y[sbert_tr_idx_all]
y_dv_all = y[sbert_dv_idx_all]

sbert_train_df_all = build_pairs_from_students(
    df_sbert_tr_all, y_tr_all,
    text_cols=TEXT_COLS,
    n_pairs_per_q=350,
    seed=SEED + 999,
    hard_frac=0.5
)
sbert_dev_df_all = build_pairs_from_students(
    df_sbert_dv_all, y_dv_all,
    text_cols=TEXT_COLS,
    n_pairs_per_q=150,
    seed=SEED + 1999,
    hard_frac=0.5
)

train_samples_all = to_input_examples(sbert_train_df_all)
dev_samples_all   = to_input_examples(sbert_dev_df_all)


# ---------
# 2) Fine-tune SBERT on ALL data
# ---------
sbert_final = build_sbert()
train_loader_all = DataLoader(
    train_samples_all,
    batch_size=SBERT_BATCH_SIZE,
    shuffle=True,
    drop_last=False
)
train_loss_all = losses.CosineSimilarityLoss(sbert_final)

dev_evaluator_all = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples_all, name="final-all-dev"
)

warmup_steps_all = int(len(train_loader_all) * SBERT_EPOCHS * SBERT_WARMUP_FRAC)

FINAL_SBERT_OUT = os.path.join(SBERT_OUT_BASE, "final_all")
os.makedirs(FINAL_SBERT_OUT, exist_ok=True)

sbert_final.fit(
    train_objectives=[(train_loader_all, train_loss_all)],
    epochs=SBERT_EPOCHS,
    warmup_steps=warmup_steps_all,
    evaluator=dev_evaluator_all,
    evaluation_steps=max(50, len(train_loader_all)//2),
    output_path=FINAL_SBERT_OUT,
    save_best_model=True,
    show_progress_bar=True,
    use_amp=torch.cuda.is_available()
)

# Reload best checkpoint
sbert_final = SentenceTransformer(FINAL_SBERT_OUT)

# ---------
# 3) Encode ALL students using final SBERT
# ---------
x_all = encode_students(sbert_final, df_students)   # (N, 9, dim)
emb_dim_final = x_all.shape[-1]
print(f"Final: embeddings shape all={x_all.shape} (dim={emb_dim_final})")

# ---------
# 4) Train final regressor head on ALL data
#    (no test set here; CV already provided generalization estimates)
# ---------
tf.keras.backend.clear_session()
reg_final = build_regressor(input_dim=emb_dim_final)

history_final = reg_final.fit(
    x_all, y,   # y is normalized 0..1
    batch_size=REG_BATCH_SIZE,
    epochs=REG_EPOCHS,
    validation_split=0.1,  # keep for early stopping; not used for reporting performance
    callbacks=[make_early_stop()],
    verbose=0
)

# ---------
# 5) Save final regressor
# ---------
FINAL_REG_OUT = "/content/regressor_final.h5"
reg_final.save(FINAL_REG_OUT)
files.download(FINAL_REG_OUT)
print(f"Saved final SBERT to: {FINAL_SBERT_OUT}")
print(f"Saved final regressor to: {FINAL_REG_OUT}")
