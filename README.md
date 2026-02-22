# DSIQ-GEN: Automatic Generation and Classification of Data Science Interview Questions

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Wiley_StatsRef-red.svg)](https://doi.org/10.1002/ISBN.stat00999.pub9)

> **DSIQ-GEN** is a three-stage pipeline for automatic generation, classification, and clustering of domain-specific data science interview questions using Parameter-Efficient Fine-Tuning (PEFT) methods on Llama-3.2-1B.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Question Generation](#question-generation)
  - [Question Classification](#question-classification)
  - [Clustering Analysis](#clustering-analysis)
- [PEFT Methods](#peft-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## 🧠 Overview

DSIQ-GEN addresses the challenge of data scarcity in specialized technical domains by combining:

1. **Question Generation** — Fine-tuning Llama-3.2-1B with LoRA, P-tuning, and Prefix tuning to generate data science interview questions conditioned on difficulty level and topic.
2. **Question Classification** — Classifying questions by domain validity, difficulty (Beginner / Intermediate / Advanced), and topic (9 categories) using LSTM, Transformer, and Feedforward neural architectures.
3. **Clustering Analysis** — Unsupervised K-means clustering with TF-IDF and Bag-of-Words representations to validate synthetic question quality and discover latent topic structure.

The framework expands a 167-question seed corpus to **1,011 questions (505.4% increase)** while maintaining semantic integrity and achieving balanced class distributions.

---

## 🏆 Key Results

| Task | Model | Dataset | Accuracy |
|------|-------|---------|----------|
| Difficulty Classification | Transformer + Keyword Tokens | Extended | **87.7%** |
| Topic Classification | Transformer + Keyword Tokens | Extended | **96.7%** |
| Clustering Purity | TF-IDF (raw) | Extended | **77.3%** |
| Question Generation (DS Rate) | LoRA CP-5 | — | **87.1%** |
| Question Generation (Diversity) | LoRA CP-5 | — | **85.0%** |

**Key Findings:**
- LoRA achieves optimal PEFT performance with only **0.9% trainable parameters**
- Conditioning fails below **8–10% class representation** threshold
- Curated **198-token keyword vocabulary** matches 357-token full lexicon with **15–18% faster training**
- **6D PCA** projections preserve ~75% of semantic variance for efficient visualization

---

## 📁 Repository Structure

```
DSIQ-GEN/
│
├── data/
│   ├── original_dataset.csv          # 167 manually curated questions
│   ├── difficulty_extended.csv       # 380 questions (difficulty-balanced)
│   ├── topic_extended.csv            # 818 questions (topic-balanced)
│   └── merged_extended.csv           # 1,011 questions (combined)
│
├── generation/
│   ├── lora_general.py               # General LoRA fine-tuning
│   ├── lora_difficulty.py            # Difficulty-conditioned LoRA
│   ├── lora_topic.py                 # Topic-conditioned LoRA
│   ├── ptuning_general.py            # P-tuning fine-tuning
│   ├── prefix_tuning_general.py      # Prefix tuning fine-tuning
│   └── generate_questions.py         # Inference / question generation script
│
├── classification/
│   ├── lstm_classifier.py            # LSTM-based classifier
│   ├── transformer_classifier.py     # Transformer encoder classifier
│   ├── feedforward_classifier.py     # Feedforward (BoW/TF-IDF) classifier
│   ├── domain_classifier.py          # Rule-based domain (DS/non-DS) classifier
│   └── train_classifiers.py          # Training pipeline for all configurations
│
├── clustering/
│   ├── kmeans_clustering.py          # K-means clustering with cosine distance
│   ├── pca_reduction.py              # PCA dimensionality reduction
│   └── evaluate_clustering.py        # Purity, recall, and visualization
│
├── utils/
│   ├── keyword_list.py               # 198 curated data science keywords
│   ├── evaluation_metrics.py         # Diversity, Uniqueness, Similarity, DS Rate, RCA
│   ├── text_representations.py       # Tokenization, BoW, TF-IDF encoders
│   └── data_utils.py                 # Dataset loading and preprocessing
│
├── notebooks/
│   ├── generation_experiments.ipynb  # Question generation experiments
│   ├── classification_experiments.ipynb # Classification experiments
│   └── clustering_analysis.ipynb     # Clustering and visualization
│
├── checkpoints/                      # Saved PEFT model checkpoints
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

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
pip install -r requirements.txt
```

### Dependencies

```
tensorflow==2.12
scikit-learn==1.2
transformers==4.30
peft==0.5
torch>=2.0
pandas==1.5
nltk==3.8
matplotlib==3.7
numpy==1.24
sentence-transformers
imbalanced-learn
```

---

## 📊 Dataset

The core dataset comprises **167 manually curated data science interview questions** annotated with:

- **Difficulty labels:** Beginner (24%), Intermediate (68.3%), Advanced (7.8%)
- **Topic labels (9 categories):**
  - Classification, Feature Selection, Neural Networks
  - Recommender Systems, Regularization, Supervised Learning
  - Text Classification, Time Series, Unsupervised Learning

Extended datasets are generated via conditioned PEFT models and stored in `data/`.

---

## 🚀 Usage

### Question Generation

**General (unconditioned) generation:**
```python
from generation.generate_questions import generate_new_question
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = PeftModel.from_pretrained(base_model, "checkpoints/lora_general/checkpoint-5")
question = generate_new_question(model, tokenizer)
print(question)
```

**Difficulty-conditioned generation:**
```python
# Fine-tune on a difficulty subset
python generation/lora_difficulty.py --difficulty beginner

# Generate beginner-level questions
python generation/generate_questions.py \
    --checkpoint checkpoints/lora_beginner/checkpoint-6 \
    --num_questions 100
```

**Topic-conditioned generation:**
```python
# Fine-tune on a topic subset
python generation/lora_topic.py --topic "time_series"

# Generate topic-specific questions
python generation/generate_questions.py \
    --checkpoint checkpoints/lora_time_series/checkpoint-3 \
    --num_questions 100
```

**Generation hyperparameters (used across all models):**
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

---

### Question Classification

**Train all classifier configurations:**
```bash
python classification/train_classifiers.py \
    --dataset data/merged_extended.csv \
    --task difficulty \
    --architecture transformer \
    --representation keyword
```

**Available options:**
| Argument | Options |
|----------|---------|
| `--task` | `domain`, `difficulty`, `topic` |
| `--architecture` | `lstm`, `transformer`, `feedforward` |
| `--representation` | `full_token`, `keyword`, `bow`, `tfidf` |
| `--dataset` | path to any CSV dataset |

**Domain classification (rule-based):**
```python
from classification.domain_classifier import is_data_science_question

question = "What is the difference between L1 and L2 regularization?"
result = is_data_science_question(question)  # True/False
```

---

### Clustering Analysis

```bash
# Run K-means clustering on extended dataset
python clustering/kmeans_clustering.py \
    --dataset data/topic_extended.csv \
    --vectorization tfidf \
    --n_clusters 9 \
    --pca_dims 6

# Evaluate clustering quality
python clustering/evaluate_clustering.py \
    --dataset data/topic_extended.csv \
    --vectorization tfidf
```

---

## 🔧 PEFT Methods

Three parameter-efficient fine-tuning approaches are implemented on **Llama-3.2-1B**:

| Method | Trainable Params | % of Total | Key Configuration |
|--------|-----------------|------------|-------------------|
| **LoRA** | 11,272,192 | 0.90% | r=6, α=32, dropout=0, all-linear modules |
| **P-tuning** | 1,024,000 | 0.08% | 500 virtual tokens via LSTM soft prompts |
| **Prefix Tuning** | 1,441,792 | 0.12% | 88 prefix tokens at each layer input |

> ✅ **LoRA is recommended** for domain-specific generation requiring both diversity and semantic fidelity.

---

## 📏 Evaluation Metrics

Five automated metrics assess generated question quality:

| Metric | Formula | Target |
|--------|---------|--------|
| **Diversity** | \|Q_unique\| / \|Q_total\| | ≥ 70% |
| **Uniqueness** | \|Q_novel\| / \|Q_unique\| × 100% | ≥ 70% |
| **Similarity** | Mean max cosine sim to training set | < 0.70 |
| **DS Rate** | % classified as valid DS questions | ≥ 80% |
| **Right Class Accuracy (RCA)** | % matching conditioning target | ≥ 70% |

Sentence embeddings are computed using `all-MiniLM-L6-v2` from the `sentence-transformers` library.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{easin2025dsiqgen,
  title     = {DSIQ-GEN: Automatic Generation and Classification of Data Science Interview Questions},
  author    = {Easin, Arafat Md},
  journal   = {Wiley StatsRef: Statistics Reference Online},
  doi       = {10.1002/ISBN.stat00999.pub9},
  year      = {2025},
  publisher = {Wiley}
}
```

---

## 🙏 Acknowledgements

This research was supported by the **Stipendium Hungaricum scholarship**, generously provided by the Hungarian Government through the Tempus Public Foundation. We also thank the **Department of Data Science at Eötvös Loránd University (ELTE)** for their academic guidance and technical support.

---

## 📬 Contact

**Arafat Md Easin**
Doctoral Fellow, Data Science and Engineering Department
Faculty of Informatics, Eötvös Loránd University (ELTE)
Budapest, Hungary
📧 arafatmdeasin@inf.elte.hu

---

<p align="center">Made with ❤️ at ELTE, Budapest</p>
