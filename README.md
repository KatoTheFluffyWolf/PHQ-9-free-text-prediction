# PHQ-9 Free-Text Depression Score Prediction

This repository contains the dataset, implementation code, trained regressor, baseline experiments, ablation experiments, robustness analysis, and interpretability scripts for the study:

**Predicting PHQ-9 Depression Scores from Free-Text Responses Using a Hybrid SBERT and Bi-LSTM Architecture**

The project explores whether short free-text responses to the nine PHQ-9 questionnaire items can be used to predict a continuous PHQ-9 total score and its corresponding severity category. The proposed architecture combines a MentalBERT-based Sentence-Transformers encoder with a Bi-LSTM regression head.

> **Important:** This repository is for academic research and non-diagnostic screening experiments only. It is not a medical device, does not provide a clinical diagnosis, and must not replace assessment or support from qualified professionals.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Main Results](#main-results)
- [Baseline Models](#baseline-models)
- [Ablation Experiments](#ablation-experiments)
- [Robustness Analysis](#robustness-analysis)
- [Interpretability Analysis](#interpretability-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility Notes](#reproducibility-notes)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

The Patient Health Questionnaire-9 (PHQ-9) is commonly administered using fixed categorical response options. This project investigates a complementary NLP-based approach in which each participant provides short free-text responses to the nine PHQ-9 items.

The proposed pipeline works as follows:

1. Each PHQ-9 item response is encoded using a Sentence-Transformers model.
2. The Sentence-Transformers encoder uses **MentalBERT** as its transformer backbone.
3. The encoder is fine-tuned using weakly supervised response pairs derived only from the training split of each cross-validation fold.
4. The nine item-level embeddings are stacked into a sequence of shape `9 × 768`.
5. A **Bi-LSTM regressor** predicts the final continuous PHQ-9 total score.
6. The predicted score is mapped to standard PHQ-9 severity categories for classification-style evaluation.

The goal is not to replace the standard PHQ-9 questionnaire. Instead, this repository provides a research prototype for studying how natural-language symptom descriptions can be modeled in a PHQ-9-style screening context.

---

## Repository Structure

```text
PHQ-9-free-text-prediction/
│
├── ablation/
│   └── scripts/files for ablation experiments
│
├── baselines/
│   └── scripts/files for baseline model comparisons
│
├── PHQ9_Student_Depression_Dataset_Aligned.xlsx
├── main.py
├── model-intepretability.py
├── regressor_final.h5
└── robustness-analysis.py
```

> Note: The interpretability script is currently named `model-intepretability.py`. For clarity, you may want to rename it to `model-interpretability.py` in a later cleanup commit.

---

## Dataset

This repository uses an aligned version of the **PHQ-9 Student Depression Dataset**.

The dataset contains:

- **250 student samples**
- **9 free-text responses per participant**, corresponding to the nine PHQ-9 questionnaire items
- **PHQ-9 total score labels** ranging from 0 to 27
- **Severity-level labels** derived from the PHQ-9 score range

The aligned dataset file included in this repository is:

```text
PHQ9_Student_Depression_Dataset_Aligned.xlsx
```

The aligned version corrects within-sample ordering issues so that each response is matched with the appropriate PHQ-9 item.

### Severity Category Mapping

| PHQ-9 Score Range | Severity Category |
|---:|---|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–27 | Severe |

### Label Note

The dataset provides pre-calculated PHQ-9 total scores and severity labels. However, the raw item-level Likert responses are not available in the dataset, and the exact label-generation procedure cannot be independently verified from the public dataset documentation. Therefore, the score labels should be interpreted as **dataset-provided PHQ-9 score labels**, not independently established clinical ground truth.

---

## Model Architecture

The proposed architecture contains two main components:

### 1. MentalBERT-based SBERT Encoder

The sentence encoder is implemented using the Sentence-Transformers framework with **MentalBERT** as the transformer backbone.

Each free-text response is converted into a fixed-dimensional sentence embedding:

```text
response → SBERT encoder → 768-dimensional embedding
```

For each participant, the nine item-level embeddings are stacked in PHQ-9 item order:

```text
E = [e1, e2, ..., e9] ∈ R^(9 × 768)
```

This preserves the questionnaire structure and allows the downstream model to process cross-item patterns.

### 2. Bi-LSTM Regressor

The sequence of nine embeddings is passed into a Bi-LSTM regression head. The Bi-LSTM models dependencies across the nine PHQ-9 item responses and outputs a continuous predicted PHQ-9 score.

The predicted score is then mapped to a severity category using the standard PHQ-9 score ranges.

---

## Main Results

The proposed MentalBERT-SBERT + Bi-LSTM model was evaluated using stratified 5-fold cross-validation.

| Metric | Result |
|---|---:|
| MAE | 1.59 ± 0.12 |
| RMSE | 1.93 ± 0.12 |
| Spearman’s ρ | 0.9559 ± 0.0043 |
| Accuracy | 97.6% ± 2.19% |
| Macro F1 | 97.14% ± 2.63% |
| Weighted F1 | 97.56% ± 2.23% |
| QWK | 0.9939 ± 0.0055 |

These results indicate strong internal cross-validation performance on the studied dataset. They should not be interpreted as evidence of broad real-world generalizability without external validation.

---

## Baseline Models

The proposed architecture was compared against several sentence embedding and transformer-based baselines, including:

- E5
- SimCSE
- MiniLM
- MPNet
- BioBERT
- BiomedBERT
- MentalBERT

The MentalBERT-based model achieved the lowest MAE and highest severity-category accuracy among the evaluated models.

### Regression Baseline Summary

| Model | MAE ↓ | RMSE | Spearman’s ρ |
|---|---:|---:|---:|
| E5 | 1.88 ± 0.20 | 2.37 ± 0.32 | 0.9448 ± 0.0092 |
| SimCSE | 1.85 ± 0.30 | 2.32 ± 0.38 | 0.9470 ± 0.0090 |
| MiniLM | 1.78 ± 0.10 | 2.24 ± 0.23 | 0.9555 ± 0.0031 |
| MPNet | 1.70 ± 0.13 | 2.16 ± 0.24 | 0.9758 ± 0.0116 |
| BioBERT | 1.64 ± 0.06 | 2.03 ± 0.12 | 0.9561 ± 0.0060 |
| BiomedBERT | 1.61 ± 0.08 | 1.95 ± 0.05 | 0.9580 ± 0.0028 |
| MentalBERT | **1.59 ± 0.12** | **1.93 ± 0.12** | 0.9559 ± 0.0043 |

---

## Ablation Experiments

The repository includes ablation experiments that test the contribution of each major component of the proposed architecture.

| Variant | Description |
|---|---|
| Variant 1 | Replace SBERT with non-transformer GloVe embeddings |
| Variant 2 | Replace MentalBERT with general-purpose BERT |
| Variant 3 | Replace Bi-LSTM with an MLP regressor |
| Variant 4 | Remove task-specific SBERT fine-tuning |
| Base | MentalBERT-based SBERT + Bi-LSTM |

The ablation results suggest that the contextual sentence encoder contributes strongly to regression accuracy, while the Bi-LSTM head improves the stability of severity classification.

### Ablation Summary

| Variant | MAE | Spearman’s ρ | Weighted F1 | QWK |
|---|---:|---:|---:|---:|
| Variant 1 | 2.19 ± 0.34 | 0.9226 ± 0.0207 | 80.00 ± 8.18 | 0.9474 ± 0.0265 |
| Variant 2 | 1.70 ± 0.17 | 0.9557 ± 0.0081 | 93.35 ± 2.39 | 0.9836 ± 0.0060 |
| Variant 3 | 2.00 ± 0.27 | 0.9522 ± 0.0040 | 74.30 ± 9.05 | 0.9430 ± 0.0197 |
| Variant 4 | 1.81 ± 0.24 | 0.9510 ± 0.0071 | 90.96 ± 2.01 | 0.9770 ± 0.0056 |
| Base | **1.59 ± 0.12** | **0.9559 ± 0.0043** | **97.56 ± 2.23** | **0.9939 ± 0.0055** |

---

## Robustness Analysis

The script `robustness-analysis.py` evaluates the model under controlled input perturbations. These experiments examine whether the trained model remains stable under noisy, incomplete, or wording-shifted inputs.

The perturbation conditions include:

- **Missing-1:** one PHQ-9 item response is replaced with an empty string
- **Missing-3:** three PHQ-9 item responses are replaced with empty strings
- **Light typo:** minor character-level noise is introduced
- **Moderate typo:** stronger character-level noise is introduced
- **Paraphrase:** selected words or phrases are replaced with semantically similar alternatives using a rule-based substitution dictionary

### Robustness Summary

| Condition | MAE | RMSE | Spearman’s ρ | QWK | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Clean | 1.527 ± 0.097 | 1.878 ± 0.110 | 0.9563 ± 0.0046 | 0.9939 ± 0.0055 | 0.9756 ± 0.0223 |
| Missing-1 | 1.622 ± 0.141 | 2.031 ± 0.209 | 0.9556 ± 0.0026 | 0.9885 ± 0.0081 | 0.9549 ± 0.0315 |
| Missing-3 | 2.301 ± 0.347 | 3.052 ± 0.538 | 0.9378 ± 0.0119 | 0.9328 ± 0.0268 | 0.7869 ± 0.0626 |
| Light typo | 1.770 ± 0.167 | 2.225 ± 0.227 | 0.9556 ± 0.0025 | 0.9840 ± 0.0067 | 0.9374 ± 0.0255 |
| Moderate typo | 2.495 ± 0.265 | 3.142 ± 0.327 | 0.9471 ± 0.0022 | 0.9341 ± 0.0097 | 0.7587 ± 0.0323 |
| Paraphrase | 1.556 ± 0.092 | 1.914 ± 0.106 | 0.9568 ± 0.0052 | 0.9941 ± 0.0057 | 0.9765 ± 0.0224 |

The results show that the model remains relatively stable under light missingness, light typo noise, and paraphrasing, while stronger missingness and moderate typo noise cause clearer performance degradation.

---

## Interpretability Analysis

The script `model-intepretability.py` provides post-hoc interpretability analyses for examining model reliance.

The included analyses are:

- **Question-level occlusion:** replaces each item embedding with a baseline embedding and measures prediction change
- **Permutation importance:** shuffles item embeddings across samples and measures performance degradation
- **Word-level saliency:** removes words or short phrases, re-encodes the response, and measures prediction change

These analyses are intended to describe how the model behaves under perturbation. They should not be interpreted as clinical explanations, causal factors, or diagnostic justification.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/KatoTheFluffyWolf/PHQ-9-free-text-prediction.git
cd PHQ-9-free-text-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn tensorflow sentence-transformers transformers torch openpyxl matplotlib seaborn scipy
```

Recommended Python version:

```text
Python 3.10+
```

If you later add a `requirements.txt`, users can install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train and Evaluate the Main Model

```bash
python main.py
```

This script runs the main MentalBERT-SBERT + Bi-LSTM training and evaluation pipeline.

### Run Robustness Analysis

```bash
python robustness-analysis.py
```

This evaluates the trained model under perturbed input conditions.

### Run Interpretability Analysis

```bash
python model-intepretability.py
```

This runs the post-hoc model-reliance analyses, including occlusion, permutation importance, and word-level saliency.

### Run Ablation Experiments

Ablation-related scripts are stored in:

```text
ablation/
```

### Run Baseline Comparisons

Baseline-related scripts are stored in:

```text
baselines/
```

---

## Reproducibility Notes

The manuscript reports the following experimental setup.

### SBERT Fine-Tuning

| Hyperparameter | Value |
|---|---:|
| Batch size | 32 |
| Warmup ratio | 0.1 |
| Epochs | 2 |

### Bi-LSTM Regressor

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | 30 |
| Batch size | 32 |
| Loss function | Mean Squared Error |
| Early stopping | Enabled |

### Evaluation Protocol

- Stratified 5-fold cross-validation
- Stratification based on PHQ-9 severity category
- Fixed hyperparameter configuration across folds
- SBERT fine-tuning pairs constructed using only the training split of each fold
- Random seed controlled across folds

### Hardware

The experiments reported in the manuscript were conducted using an NVIDIA T4 GPU.

---

## Ethical Considerations

This repository is provided for academic research purposes only.

The model:

- is not a diagnostic tool;
- should not be used for emergency mental-health assessment;
- should not replace qualified professional judgment;
- should not be used to make clinical, academic, disciplinary, administrative, or high-stakes decisions;
- requires independent external validation before any real-world use.

Any use in student-support or mental-health-related contexts should be optional, consent-based, reviewed by trained staff, and paired with appropriate referral pathways.

---

## Limitations

Several limitations should be considered:

1. The dataset contains only 250 student samples.
2. The responses are highly constrained and semi-structured, not fully open-ended clinical narratives.
3. The model has only been evaluated using internal stratified 5-fold cross-validation.
4. External validation on independent datasets has not yet been performed.
5. The dataset-provided PHQ-9 labels cannot be independently verified from raw Likert-scale responses.
6. The model should not be assumed to generalize to non-student populations, clinical patients, older adults, or culturally different groups without further validation.
7. Performance may degrade under strong noise, missing responses, or distribution shift.

---

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{nguyen2026phq9freetext,
  title   = {Predicting PHQ-9 Depression Scores from Free-Text Responses Using a Hybrid SBERT and Bi-LSTM Architecture},
  author  = {Nguyen, Duy-Anh and Nguyen, Trung-Hau and Trinh, Hoang-Khoa},
  journal = {Vietnam Journal of Computer Science},
  year    = {2026},
  note    = {Manuscript under revision}
}
```

Please also cite the original dataset source if you use the dataset in a separate study.

---

## Authors

- **Duy-Anh Nguyen**  
  Department of Computing, FPT University - Greenwich Vietnam

- **Trung-Hau Nguyen**  
  Department of Computing, FPT University - Greenwich Vietnam

- **Hoang-Khoa Trinh**  
  College of Engineering, Georgia Institute of Technology

---

## License

No license has been specified yet.

Before reuse or redistribution, please add a `LICENSE` file to clarify how the code, trained model, and dataset copy may be used.

Recommended options:

- **MIT License** for broad open-source reuse of code
- **Apache License 2.0** for open-source code with explicit patent terms
- **CC BY 4.0** for documentation or research artifacts

Dataset reuse may also depend on the license and terms of the original dataset provider.
