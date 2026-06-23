import re
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/content/PHQ9_Student_Depression_Dataset_Aligned.xlsx"
SBERT_PATH = "/content/drive/MyDrive/PHQ9/Fine-tunedSBERT"       # folder saved with SentenceTransformer.save(...)
REGRESSOR_PATH = "/content/regressor_final.h5"  # or .h5
OUTPUT_DIR = "/content/xai_outputs"

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
MODEL_OUTPUT_IS_NORMALIZED = True   # True if regressor predicts y/27; False if it predicts raw 0-27

SEED = 42
PERMUTATION_REPEATS = 30
CASE_STUDY_IDXS = [193, 39, 15]

RUN_WORD_LEVEL_SALIENCY = True
WORD_SALIENCY_TOP_HIGH_SCORE_N = 10
WORD_SALIENCY_NGRAMS = [1]          # use [1, 2, 3] for words + short phrases
WORD_SALIENCY_MAX_TOKENS = 40
WORD_SALIENCY_TOP_K = 50
PERTURBATION = "remove"             # "remove" or "mask"
MASK_TOKEN = "[MASK]"


# ============================================================
# SMALL HELPERS
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    return "" if x.lower() in {"nan", "none", "null"} else x


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def predict_scores(model, X):
    pred = model.predict(X.astype("float32"), verbose=0).reshape(-1)
    if MODEL_OUTPUT_IS_NORMALIZED:
        pred = pred * 27.0
    return np.clip(pred, 0, 27)


def metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
    }


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", clean_text(text), flags=re.UNICODE)


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def perturb(tokens, start, n, mode="remove"):
    removed = detokenize(tokens[start:start + n])
    if mode == "mask":
        changed = tokens[:start] + [MASK_TOKEN] + tokens[start + n:]
    else:
        changed = tokens[:start] + tokens[start + n:]
    return removed, detokenize(changed)


# ============================================================
# LOAD DATA / MODELS / EMBEDDINGS
# ============================================================

def load_data():
    df = pd.read_excel(DATA_PATH)
    df[QUESTION_COLS] = df[QUESTION_COLS].apply(lambda s: s.map(clean_text))
    y = pd.to_numeric(df[SCORE_COL], errors="coerce").to_numpy(dtype="float32")
    return df, y


def create_embeddings(df, sbert):
    item_embeddings = []
    for col in QUESTION_COLS:
        emb = sbert.encode(
            df[col].tolist(),
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        item_embeddings.append(emb)
    return np.stack(item_embeddings, axis=1).astype("float32")  # (N, 9, D)


# ============================================================
# METHOD 1: QUESTION-LEVEL OCCLUSION
# ============================================================

def run_occlusion(model, X, y, out_dir):
    labels = [f"Q{i}" for i in range(1, 10)]
    base_pred = predict_scores(model, X)
    mean_baseline = X.mean(axis=0)  # (9, D)

    signed = np.zeros((len(X), 9), dtype="float32")
    absolute = np.zeros((len(X), 9), dtype="float32")

    for i in range(9):
        X_occ = X.copy()
        X_occ[:, i, :] = mean_baseline[i]
        occ_pred = predict_scores(model, X_occ)
        signed[:, i] = base_pred - occ_pred
        absolute[:, i] = np.abs(signed[:, i])

    global_df = pd.DataFrame({
        "item": labels,
        "mean_abs_delta": absolute.mean(axis=0),
        "std_abs_delta": absolute.std(axis=0, ddof=1),
        "mean_signed_delta": signed.mean(axis=0),
        "std_signed_delta": signed.std(axis=0, ddof=1),
    })

    top1 = absolute.argmax(axis=1)
    top1_df = pd.DataFrame({
        "item": labels,
        "top1_count": np.bincount(top1, minlength=9),
        "top1_frequency": np.bincount(top1, minlength=9) / len(X),
    })

    global_df.to_csv(out_dir / "occlusion_global_summary.csv", index=False)
    top1_df.to_csv(out_dir / "occlusion_top1_frequency.csv", index=False)
    np.save(out_dir / "occlusion_signed_deltas.npy", signed)
    np.save(out_dir / "occlusion_abs_deltas.npy", absolute)

    plt.figure(figsize=(8, 5))
    plt.bar(global_df["item"], global_df["mean_abs_delta"])
    plt.xlabel("PHQ-9 item")
    plt.ylabel("Mean absolute prediction change |Δ|")
    plt.title("Question-level occlusion: global mean |Δ|")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_occlusion_global_mean_abs_delta.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(top1_df["item"], top1_df["top1_frequency"])
    plt.xlabel("PHQ-9 item")
    plt.ylabel("Top-1 frequency")
    plt.title("Question-level occlusion: Top-1 frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_occlusion_top1_frequency.png", dpi=300)
    plt.close()

    for idx in CASE_STUDY_IDXS:
        case_df = pd.DataFrame({
            "item": labels,
            "signed_delta": signed[idx],
            "abs_delta": absolute[idx],
        })
        case_df.to_csv(out_dir / f"occlusion_case_idx_{idx}.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(case_df["item"], case_df["abs_delta"])
        plt.xlabel("PHQ-9 item")
        plt.ylabel("Absolute prediction change |Δ|")
        plt.title(f"Question-level occlusion idx={idx} | true={y[idx]:.2f}, pred={base_pred[idx]:.2f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_occlusion_case_idx_{idx}.png", dpi=300)
        plt.close()

    return global_df, top1_df


# ============================================================
# METHOD 2: QUESTION-LEVEL PERMUTATION IMPORTANCE
# ============================================================

def run_permutation_importance(model, X, y, out_dir):
    labels = [f"Q{i}" for i in range(1, 10)]
    rng = np.random.default_rng(SEED)

    base_pred = predict_scores(model, X)
    base = metrics(y, base_pred)
    rows, repeat_rows = [], []

    for i in range(9):
        delta_mae, delta_rmse = [], []

        for r in range(PERMUTATION_REPEATS):
            X_perm = X.copy()
            X_perm[:, i, :] = X[rng.permutation(len(X)), i, :]
            pred = predict_scores(model, X_perm)
            m = metrics(y, pred)

            d_mae = m["MAE"] - base["MAE"]
            d_rmse = m["RMSE"] - base["RMSE"]
            delta_mae.append(d_mae)
            delta_rmse.append(d_rmse)

            repeat_rows.append({
                "item": labels[i],
                "repeat": r,
                "delta_mae": d_mae,
                "delta_rmse": d_rmse,
                "mae_perm": m["MAE"],
                "rmse_perm": m["RMSE"],
            })

        rows.append({
            "item": labels[i],
            "baseline_mae": base["MAE"],
            "baseline_rmse": base["RMSE"],
            "mean_delta_mae": float(np.mean(delta_mae)),
            "std_delta_mae": float(np.std(delta_mae, ddof=1)),
            "mean_delta_rmse": float(np.mean(delta_rmse)),
            "std_delta_rmse": float(np.std(delta_rmse, ddof=1)),
        })

    summary = pd.DataFrame(rows)
    pd.DataFrame(repeat_rows).to_csv(out_dir / "permutation_importance_all_repeats.csv", index=False)
    summary.to_csv(out_dir / "permutation_importance_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary["item"], summary["mean_delta_rmse"], yerr=summary["std_delta_rmse"], capsize=4)
    plt.xlabel("PHQ-9 item")
    plt.ylabel("ΔRMSE")
    plt.title(f"Permutation importance: ΔRMSE over {PERMUTATION_REPEATS} repeats")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_permutation_delta_rmse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(summary["item"], summary["mean_delta_mae"], yerr=summary["std_delta_mae"], capsize=4)
    plt.xlabel("PHQ-9 item")
    plt.ylabel("ΔMAE")
    plt.title(f"Permutation importance: ΔMAE over {PERMUTATION_REPEATS} repeats")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_permutation_delta_mae.png", dpi=300)
    plt.close()

    return summary


# ============================================================
# METHOD 3: WORD-LEVEL OCCLUSION SALIENCY
# ============================================================

def run_word_saliency(model, sbert, df, X, y, out_dir):
    sample_idxs = np.argsort(-y)[:WORD_SALIENCY_TOP_HIGH_SCORE_N].tolist()
    rows = []

    for idx in sample_idxs:
        x0 = X[idx:idx + 1].copy()
        base_pred = float(predict_scores(model, x0)[0])

        for q_idx, col in enumerate(QUESTION_COLS):
            tokens = tokenize(df.iloc[idx][col])[:WORD_SALIENCY_MAX_TOKENS]
            if not tokens:
                continue

            removed_spans, perturbed_texts, starts, n_sizes = [], [], [], []
            for n in WORD_SALIENCY_NGRAMS:
                for start in range(len(tokens) - n + 1):
                    removed, changed = perturb(tokens, start, n, PERTURBATION)
                    removed_spans.append(removed)
                    perturbed_texts.append(changed)
                    starts.append(start)
                    n_sizes.append(n)

            pert_embs = sbert.encode(
                perturbed_texts,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

            X_batch = np.repeat(x0, len(perturbed_texts), axis=0)
            X_batch[:, q_idx, :] = pert_embs
            pert_preds = predict_scores(model, X_batch)

            for removed, changed, start, n, pert_pred in zip(
                removed_spans, perturbed_texts, starts, n_sizes, pert_preds
            ):
                delta = base_pred - float(pert_pred)
                rows.append({
                    "sample_idx": idx,
                    "true_score": float(y[idx]),
                    "base_prediction": base_pred,
                    "question": f"Q{q_idx + 1}",
                    "column": col,
                    "ngram_size": n,
                    "token_start": start,
                    "perturbation": PERTURBATION,
                    "removed_text": removed,
                    "perturbed_text": changed,
                    "perturbed_prediction": float(pert_pred),
                    "delta_score": delta,
                    "abs_delta_score": abs(delta),
                })

    sal = pd.DataFrame(rows)
    sal.to_csv(out_dir / "word_level_saliency_all.csv", index=False)

    top_pos = sal.sort_values("delta_score", ascending=False).head(WORD_SALIENCY_TOP_K)
    top_neg = sal.sort_values("delta_score", ascending=True).head(WORD_SALIENCY_TOP_K)
    top_abs = sal.sort_values("abs_delta_score", ascending=False).head(WORD_SALIENCY_TOP_K)

    top_pos.to_csv(out_dir / "word_level_saliency_top_positive.csv", index=False)
    top_neg.to_csv(out_dir / "word_level_saliency_top_negative.csv", index=False)
    top_abs.to_csv(out_dir / "word_level_saliency_top_absolute.csv", index=False)

    by_token = (
        sal.groupby(["removed_text", "ngram_size"], as_index=False)
        .agg(
            mean_delta_score=("delta_score", "mean"),
            mean_abs_delta_score=("abs_delta_score", "mean"),
            max_abs_delta_score=("abs_delta_score", "max"),
            count=("delta_score", "size"),
        )
        .sort_values("mean_abs_delta_score", ascending=False)
    )
    by_token.to_csv(out_dir / "word_level_saliency_by_token.csv", index=False)

    by_question = (
        sal.groupby(["question", "column"], as_index=False)
        .agg(
            mean_delta_score=("delta_score", "mean"),
            mean_abs_delta_score=("abs_delta_score", "mean"),
            max_abs_delta_score=("abs_delta_score", "max"),
            n_perturbations=("delta_score", "size"),
        )
        .sort_values("question")
    )
    by_question.to_csv(out_dir / "word_level_saliency_by_question.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(by_question["question"], by_question["mean_abs_delta_score"])
    plt.xlabel("PHQ-9 item")
    plt.ylabel("Mean absolute word-level Δ")
    plt.title("Word-level saliency by occlusion: mean |Δ| by item")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_word_level_saliency_by_question.png", dpi=300)
    plt.close()

    plot_df = by_token.head(15).copy()
    plot_df["label"] = plot_df["removed_text"].astype(str).str.slice(0, 25)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["label"][::-1], plot_df["mean_abs_delta_score"][::-1])
    plt.xlabel("Mean absolute word-level Δ")
    plt.ylabel("Removed word/phrase")
    plt.title("Top word/phrase saliency by occlusion")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_word_level_saliency_top_tokens.png", dpi=300)
    plt.close()

    return sal


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, y = load_data()
    sbert = SentenceTransformer(SBERT_PATH)
    model = tf.keras.models.load_model(REGRESSOR_PATH, compile=False)

    print("Encoding dataset with fine-tuned SBERT...")
    X = create_embeddings(df, sbert)
    np.save(out_dir / "created_embeddings.npy", X)
    print("Embedding shape:", X.shape)

    pred = predict_scores(model, X)
    base = metrics(y, pred)
    print("Baseline metrics:", base)

    pd.DataFrame({
        "id": df["ID"],
        "true_score": y,
        "predicted_score": pred,
        "error": pred - y,
        "abs_error": np.abs(pred - y),
    }).to_csv(out_dir / "baseline_predictions.csv", index=False)

    print("Running question-level occlusion...")
    occ_global, occ_top1 = run_occlusion(model, X, y, out_dir)
    print(occ_global)
    print(occ_top1)

    print("Running permutation importance...")
    perm = run_permutation_importance(model, X, y, out_dir)
    print(perm)

    if RUN_WORD_LEVEL_SALIENCY:
        print("Running word-level saliency...")
        sal = run_word_saliency(model, sbert, df, X, y, out_dir)
        print("Word-level saliency rows:", len(sal))

    print("Done. Outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
