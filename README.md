# DSIQ-GEN: Automatic Generation and Classification of Data Science Interview Questions

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/arafatro/DSIQ-GEN/blob/main/LICENSE)

> **DSIQ-GEN** is a reproducible research repository for generation, classification, and clustering of data science interview questions.
>
> The associated article is currently under review in the *International Journal of Intelligent Systems*.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Run the Repository](#run-the-repository)
- [Usage](#usage)
  * [Question Generation](#question-generation)
  * [Question Classification](#question-classification)
  * [Clustering Analysis](#clustering-analysis)
- [PEFT Methods](#peft-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

DSIQ-GEN addresses the challenge of data scarcity in specialized technical domains by combining:

1. **Question Generation** — Fine-tuning Llama-3.2-1B with LoRA, P-tuning, and Prefix tuning to generate data science interview questions conditioned on difficulty level and topic.
2. **Question Classification** — Classifying questions by domain validity, difficulty (Beginner / Intermediate / Advanced), and topic (9 categories) using LSTM, Transformer, and Feedforward neural architectures.
3. **Clustering Analysis** — Unsupervised K-means clustering with TF-IDF and Bag-of-Words representations to validate synthetic question quality and discover latent topic structure.

The framework expands a 167-question seed corpus to **1,011 questions (505.4% increase)** while maintaining semantic integrity and achieving balanced class distributions.

---

## Key Results

| Task                            | Model                        | Dataset  | Accuracy  |
| -------------------------------- | ----------------------------- | -------- | --------- |
| Difficulty Classification        | Transformer + Keyword Tokens  | Extended | **87.7%** |
| Topic Classification             | Transformer + Keyword Tokens  | Extended | **96.7%** |
| Clustering Purity                | TF-IDF (raw)                  | Extended | **77.3%** |
| Question Generation (DS Rate)    | LoRA CP-5                     | —        | **87.1%** |
| Question Generation (Diversity)  | LoRA CP-5                     | —        | **85.0%** |

**Key Findings:**

- LoRA achieves optimal PEFT performance with only **0.9% trainable parameters**
- Conditioning fails below **8–10% class representation** threshold
- Curated **198-token keyword vocabulary** matches 357-token full lexicon with **15–18% faster training**
- **6D PCA** projections preserve ~75% of semantic variance for efficient visualization

---

## Repository Structure

```
DSIQ-GEN/
│
├── data/
│   ├── dataset_6.csv
│   ├── dataset_extended_difficulty.csv
│   ├── dataset_extended_merged.csv
│   ├── dataset_extended_merged_data_science_only.csv
│   ├── dataset_extended_topic.csv
│   ├── dataset_non_data_science.csv
│   └── question_dataset.csv
│
├── classification models/            # Saved trained classifier weights
│   ├── difficulty/
│   └── topic/
│
├── generated text/                   # Generated question outputs
│   ├── for evaluation/               # Per-checkpoint generation samples
│   └── for extended datasets/        # Final generated pools used to build data/
│
├── notebooks/                        # All experiment code (generation, classification, clustering)
│   ├── LoRA_models_for_text_generation.ipynb
│   ├── P_tuning_for_text_generation.ipynb
│   ├── Prefix_tuning_for_text_generation.ipynb
│   ├── Evaluation_+_Extended_dataset_creation.ipynb
│   ├── Classification.ipynb
│   └── Clustering.ipynb
│
├── environment.txt                   # Python dependency specifications
├── LICENSE                           # MIT license for the repository
├── CITATION.md
├── keywords.txt
├── question generation models.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11
- NVIDIA GPU with ≥16GB VRAM (tested on RTX 4060 Ti)
- CUDA-compatible environment

### Setup

```bash
# Clone the repository
git clone https://github.com/arafatro/DSIQ-GEN.git
cd DSIQ-GEN

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r environment.txt
```

This repository is licensed under the MIT License. See `LICENSE` for details.

### Dependencies

```
tensorflow==2.12.0
scikit-learn==1.2.2
transformers==4.30.2
peft==0.5.0
torch>=2.0.0
pandas==1.5.3
nltk==3.8.1
matplotlib==3.7.1
numpy==1.24.3
sentence-transformers==2.2.2
imbalanced-learn==0.11.0
```

---

## Dataset

All CSV datasets are in the `data/` folder.

The core dataset comprises **167 manually curated data science interview questions** annotated with:

- **Difficulty labels:** Beginner (24%), Intermediate (68.3%), Advanced (7.8%)
- **Topic labels (9 categories):** Classification, Feature Selection, Neural Networks, Recommender Systems, Regularization, Supervised Learning, Text Classification, Time Series, Unsupervised Learning

Key data files:

- `data/question_dataset.csv` — original question corpus
- `data/dataset_extended_difficulty.csv` — difficulty-balanced extended dataset
- `data/dataset_extended_topic.csv` — topic-balanced extended dataset
- `data/dataset_extended_merged.csv` — merged dataset combining both extensions
- `data/dataset_extended_merged_data_science_only.csv` — DS-only merged split
- `data/dataset_non_data_science.csv` — non-data-science examples used for domain-classifier evaluation

---

## Run the Repository

All experiments are implemented as Jupyter notebooks in `notebooks/`. There are no standalone CLI scripts; run the notebooks directly in the order below.

1. Install dependencies described above.
2. Make sure all CSV datasets are present in `data/`.
3. Open and run notebooks in this order:

| Step | Notebook | Purpose |
| ---- | -------- | ------- |
| 1 | `notebooks/LoRA_models_for_text_generation.ipynb` | Fine-tunes Llama-3.2-1B with LoRA (general, difficulty-conditioned, topic-conditioned) |
| 2 | `notebooks/P_tuning_for_text_generation.ipynb` | Fine-tunes with P-tuning |
| 3 | `notebooks/Prefix_tuning_for_text_generation.ipynb` | Fine-tunes with Prefix tuning |
| 4 | `notebooks/Evaluation_+_Extended_dataset_creation.ipynb` | Computes generation metrics (Diversity, Uniqueness, Similarity, DS Rate, RCA) and builds the extended datasets in `data/` |
| 5 | `notebooks/Classification.ipynb` | Trains and evaluates all 25 difficulty/topic classifier configurations; saves weights to `classification models/` |
| 6 | `notebooks/Clustering.ipynb` | Runs K-means clustering (BoW/TF-IDF, with/without PCA) and computes purity/recall |

Trained classifier weights (`.h5`) are already included in `classification models/`, so `Classification.ipynb` and `Clustering.ipynb` can be run for evaluation without retraining from scratch.

---

## Usage

### Question Generation

Run `notebooks/LoRA_models_for_text_generation.ipynb` (or the P-tuning / Prefix-tuning equivalents) end to end. Each notebook fine-tunes the base model and saves checkpoints locally, then generates candidate questions using the shared decoding configuration:

```python
generate_kwargs = {
    "num_beams": 3,
    "temperature": 1.5,
    "top_p": 0.75,
    "repetition_penalty": 2.0,
    "min_new_tokens": 5,
    "max_new_tokens": 50
}
```

Raw generated outputs used for checkpoint evaluation and dataset extension are already provided in `generated text/for evaluation/` and `generated text/for extended datasets/`, so generation does not need to be rerun to reproduce downstream classification/clustering results.

### Question Classification

Run `notebooks/Classification.ipynb`. The notebook trains all combinations of task (`domain`, `difficulty`, `topic`), architecture (`lstm`, `transformer`, `feedforward`), representation (`full_token`, `keyword`, `bow`, `tfidf`), and dataset variant (original, difficulty-extended, topic-extended, merged), and saves weights to `classification models/`.

Domain classification is rule-based (Equation 1 in the paper) and does not require a trained model; the implementation is in the same notebook.

### Clustering Analysis

Run `notebooks/Clustering.ipynb`. The notebook performs K-means clustering (k=9) on BoW and TF-IDF vectorizations, with and without PCA (3D/6D/15D), and reports purity and recall against the manually assigned topic labels.

---

## PEFT Methods

Three parameter-efficient fine-tuning approaches are implemented on **Llama-3.2-1B**:

| Method            | Trainable Params | % of Total | Key Configuration                        |
| ----------------- | ----------------- | ---------- | ----------------------------------------- |
| **LoRA**          | 11,272,192        | 0.90%      | r=6, α=32, dropout=0, all-linear modules |
| **P-tuning**      | 1,024,000          | 0.08%      | 500 virtual tokens via LSTM soft prompts |
| **Prefix Tuning** | 1,441,792          | 0.12%      | 88 prefix tokens at each layer input     |

> **LoRA is recommended** for domain-specific generation requiring both diversity and semantic fidelity.

---

## Evaluation Metrics

Five automated metrics assess generated question quality:

| Metric                         | Formula                              | Target |
| -------------------------------- | ------------------------------------- | ------ |
| **Diversity**                  | \|Q_unique\| / \|Q_total\|            | ≥ 70%  |
| **Uniqueness**                 | \|Q_novel\| / \|Q_unique\| × 100%     | ≥ 70%  |
| **Similarity**                 | Mean max cosine sim to training set   | < 0.70 |
| **DS Rate**                    | % classified as valid DS questions    | ≥ 80%  |
| **Right Class Accuracy (RCA)** | % matching conditioning target        | ≥ 70%  |

Sentence embeddings are computed using `all-MiniLM-L6-v2` from the `sentence-transformers` library.

---

## Data Availability

The core and extended question datasets (167 original questions and all extended variants totaling 1,011 questions), raw generated-question outputs used for checkpoint evaluation and dataset extension, trained classifier weights for all difficulty and topic models, keyword lexicon, and analysis notebooks are all included in this repository. The fine-tuned PEFT adapter checkpoints for the question-generation models (LoRA, P-tuning, and Prefix tuning) are not included due to file size, but are fully reproducible by running the provided fine-tuning notebooks on the released 167-question seed corpus with the configurations listed in the [PEFT Methods](#peft-methods) table above.

---

## Citation

If you use this work, please cite:

```bibtex
@article{easin2025dsiqgen,
  title     = {DSIQ-GEN: Automatic Generation and Classification
               of Data Science Interview Questions},
  author    = {Easin, Arafat Md and
               Barbara, Cs{\'a}sz{\'a}r Fanni and
               Farou, Zakarya and
               Orosz, Tam{\'a}s},
  journal   = {International Journal of Intelligent Systems},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

This research was supported by the **Stipendium Hungaricum scholarship**, generously provided by the Hungarian Government through the Tempus Public Foundation. We also thank the **Department of Data Science at Eötvös Loránd University (ELTE)** for their academic guidance and technical support.

---

## Contact

**Arafat Md Easin**
Doctoral Fellow, Data Science and Engineering Department
Faculty of Informatics, Eötvös Loránd University (ELTE)
Budapest, Hungary
📧 arafatmdeasin@inf.elte.hu

---

Made with ❤️ at ELTE, Budapest
