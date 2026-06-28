# =========================
# Install (Colab)
# =========================
!pip -q install -U sentence-transformers openpyxl tensorflow transformers

import os, random
import pandas as pd
import numpy as np
import re

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    cohen_kappa_score, classification_report,
    accuracy_score
)
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# =========================
# Config
# =========================
SEED = 42
K = 5

STUDENTS_XLSX = "/content/PHQ9_Student_Depression_Dataset_Aligned.xlsx"

REG_BATCH_SIZE = 32
REG_EPOCHS = 30
REG_LR = 1e-4

# SimCSE model id (supervised recommended)
SENT_MODEL_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"
# or: "princeton-nlp/unsup-simcse-bert-base-uncased"


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# Load data
# =========================
df_students = pd.read_excel(STUDENTS_XLSX)

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

if all(q in df_students.columns for q in Questions):
    TEXT_COLS = Questions
else:
    raise ValueError(
        "Could not find raw-text columns for all 9 questions in df_students.\n"
        "Either name your columns exactly as the Questions list, or define TEXT_COLS mapping manually."
    )

# Target y (normalized 0..1)
y = (df_students["PHQ-9 Score"].values / 27.0).astype(np.float32)

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


# =========================
# SimCSE: load + encode
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
mdl = AutoModel.from_pretrained(SENT_MODEL_NAME).to(device)
mdl.eval()

print("Loaded SimCSE:", SENT_MODEL_NAME, "| device:", device)

def encode_texts_simcse(texts, batch_size=64, normalize=True):
    """
    Returns (N, dim) embeddings for a list of texts using SimCSE.
    Uses pooler_output when available, otherwise CLS token.
    """
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch = tok(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(device)

            out = mdl(**batch)

            if getattr(out, "pooler_output", None) is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0]  # CLS

            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

            all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)

def encode_students(df_sub, batch_size=64, normalize=True, show_progress=False):
    """
    Returns x of shape (n_samples, 9, emb_dim) from raw text using SimCSE.
    Encodes all 9 answers per student as a sequence of sentence embeddings.
    """
    arr = df_sub[TEXT_COLS].fillna("").astype(str).values
    n = arr.shape[0]
    flat_texts = arr.reshape(-1).tolist()

    # SimCSE encode
    embs = encode_texts_simcse(flat_texts, batch_size=batch_size, normalize=normalize)
    x = embs.reshape(n, len(TEXT_COLS), -1).astype(np.float32)
    return x


# =========================
# K-Fold CV: Encode -> regressor train/eval
# =========================
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

fold_mae, fold_rmse, fold_spearman, fold_qwk = [], [], [], []
fold_acc, fold_macro_f1, fold_weighted_f1 = [], [], []
fold_macro_recall, fold_macro_precision = [], []
fold_weighted_recall, fold_weighted_precision = [], []


for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(sev)), sev), start=1):
    print(f"\n================ Fold {fold}/{K} ================\n")
    set_seed(SEED + fold)

    df_train_students = df_students.iloc[train_idx].reset_index(drop=True)
    df_test_students  = df_students.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ---------
    # 1) Encode train/test students with SimCSE
    # ---------
    x_train = encode_students(df_train_students, batch_size=64, normalize=True)
    x_test  = encode_students(df_test_students,  batch_size=64, normalize=True)

    emb_dim = x_train.shape[-1]
    print(f"Fold {fold}: embeddings shape train={x_train.shape}, test={x_test.shape} (dim={emb_dim})")

    # ---------
    # 2) Train regressor head (fresh per fold)
    # ---------
    tf.keras.backend.clear_session()
    reg_model = build_regressor(input_dim=emb_dim)

    _ = reg_model.fit(
        x_train, y_train,
        batch_size=REG_BATCH_SIZE,
        epochs=REG_EPOCHS,
        validation_split=0.2,
        callbacks=[make_early_stop()],
        verbose=0
    )

    # ---------
    # 3) Evaluate
    # ---------
    y_pred = reg_model.predict(x_test, verbose=0).flatten()

    y_pred_den = np.clip(y_pred * 27, 0, 27)
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

    macro_precision    = report_dict["macro avg"]["precision"]
    macro_recall       = report_dict["macro avg"]["recall"]
    weighted_precision = report_dict["weighted avg"]["precision"]
    weighted_recall    = report_dict["weighted avg"]["recall"]
    macro_f1           = report_dict["macro avg"]["f1-score"]
    weighted_f1        = report_dict["weighted avg"]["f1-score"]

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
print(f"MAE     : {np.mean(fold_mae):.2f} ± {np.std(fold_mae, ddof=1):.2f}")
print(f"RMSE    : {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse, ddof=1):.2f}")
print(f"Spearman: {np.mean(fold_spearman):.4f} ± {np.std(fold_spearman, ddof=1):.4f}")
print(f"QWK     : {np.mean(fold_qwk):.4f} ± {np.std(fold_qwk, ddof=1):.4f}")

print("\n--- Severity Classification Summary ---")
print(f"Accuracy     : {np.mean(fold_acc):.4f} ± {np.std(fold_acc, ddof=1):.4f}")
print(f"Macro Precision : {np.mean(fold_macro_precision):.4f} ± {np.std(fold_macro_precision, ddof=1):.4f}")
print(f"Macro Recall    : {np.mean(fold_macro_recall):.4f} ± {np.std(fold_macro_recall, ddof=1):.4f}")
print(f"Weighted Precision: {np.mean(fold_weighted_precision):.4f} ± {np.std(fold_weighted_precision, ddof=1):.4f}")
print(f"Weighted Recall   : {np.mean(fold_weighted_recall):.4f} ± {np.std(fold_weighted_recall, ddof=1):.4f}")
print(f"Macro F1     : {np.mean(fold_macro_f1):.4f} ± {np.std(fold_macro_f1, ddof=1):.4f}")
print(f"Weighted F1  : {np.mean(fold_weighted_f1):.4f} ± {np.std(fold_weighted_f1, ddof=1):.4f}")
