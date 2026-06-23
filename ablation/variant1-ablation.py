# =========================
# Install (Colab)
# =========================
!pip -q install -U sentence-transformers openpyxl tensorflow
import os, random
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, classification_report, accuracy_score, confusion_matrix
from scipy.stats import spearmanr

import torch
from torch.utils.data import DataLoader

import re
import numpy as np
import pickle

# =========================
# Config
# =========================
SEED = 42
K = 5

STUDENTS_XLSX = "/content/PHQ9_Student_Depression_Dataset_Aligned.xlsx"

REG_BATCH_SIZE = 32
REG_EPOCHS = 30
REG_LR = 1e-4

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


# =========================
# GloVe config (assume file exists)
# =========================
GLOVE_PATH = "/content/glove.6B.300d.txt"   # <-- change to your actual glove file path (e.g., glove.6B.300d.txt)

def infer_glove_dim(glove_path):
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) > 2:
                return len(parts) - 1
    raise ValueError("Could not infer GloVe dim (file seems empty or malformed).")

GLOVE_DIM = infer_glove_dim(GLOVE_PATH)

# -------------------------
# Tokenizer (simple + robust)
# -------------------------
_token_re = re.compile(r"[a-zA-Z0-9']+")
def tokenize(text: str):
    text = str(text).lower().strip()
    if not text or text == "nan":
        return []
    return _token_re.findall(text)

# -------------------------
# Build vocab from your dataset (and optionally pairs) so we only load needed GloVe rows
# -------------------------
def build_vocab_from_data(df_students, text_cols, df_pairs=None):
    vocab = set()
    for col in text_cols:
        for t in df_students[col].astype(str).tolist():
            vocab.update(tokenize(t))

    if df_pairs is not None:
        for t in df_pairs["text1"].astype(str).tolist():
            vocab.update(tokenize(t))
        for t in df_pairs["text2"].astype(str).tolist():
            vocab.update(tokenize(t))

    return vocab

VOCAB = build_vocab_from_data(df_students, TEXT_COLS)

def make_early_stop():
    return EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

def build_regressor(input_dim):
    # (9, dim) input expected
    model = Sequential([
        Bidirectional(LSTM(128)),
        Dense(1, activation=None)
    ])
    model.compile(optimizer=Adam(learning_rate=REG_LR), loss="mse", metrics=["mae"])
    return model

def load_glove_subset(glove_path, vocab, dim):
    glove = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            w = parts[0]
            if w in vocab:
                glove[w] = np.asarray(parts[1:], dtype=np.float32)
    return glove

GLOVE = load_glove_subset(GLOVE_PATH, VOCAB, GLOVE_DIM)
print(f"Loaded {len(GLOVE):,}/{len(VOCAB):,} vocab words from GloVe (dim={GLOVE_DIM})")

# -------------------------
# Mean-pooled sentence embedding from GloVe
# -------------------------
def glove_sentence_embedding(text, glove_dict, dim):
    toks = tokenize(text)
    vecs = [glove_dict[t] for t in toks if t in glove_dict]
    if not vecs:
        return np.zeros((dim,), dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

# =========================
# REPLACEMENT: encode_students() using GloVe
# =========================
def encode_students(df_sub):
    """
    Returns x of shape (n_samples, 9, GLOVE_DIM) from raw text using GloVe mean-pooling.
    """
    emb_list = []
    for q in TEXT_COLS:
        q_texts = df_sub[q].astype(str).str.strip().tolist()
        embs = np.stack([glove_sentence_embedding(t, GLOVE, GLOVE_DIM) for t in q_texts], axis=0)
        emb_list.append(embs)  # (n, dim)

    x = np.stack(emb_list, axis=1).astype(np.float32)  # (n, 9, dim)
    return x

# =========================
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

fold_mae, fold_rmse, fold_spearman, fold_qwk = [], [], [], []
fold_acc, fold_macro_f1, fold_weighted_f1, fold_macro_recall, fold_macro_precision, fold_weighted_recall, fold_weighted_precision = [], [], [], [], [], [], []

# Create output directories


for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(sev)), sev), start=1):
    print(f"\n================ Fold {fold}/{K} ================\n")
    set_seed(SEED + fold)

    df_train_students = df_students.iloc[train_idx].reset_index(drop=True)
    df_test_students  = df_students.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ---------
    # 3) Encode train/test students
    # ---------
    x_train = encode_students(df_train_students)
    x_test  = encode_students(df_test_students)

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

    class_names = ["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"]


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
    print("\nClassification Report (Fold {})".format(fold))
    print(classification_report(
        true_cls, pred_cls,
        labels=[1,2,3,4,5],
        target_names=["1-Minimal", "2-Mild", "3-Moderate", "4-Mod. Severe", "5-Severe"],
        digits=4,
        zero_division=0
    ))
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

# =========================
