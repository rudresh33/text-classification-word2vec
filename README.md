# Text Classification Ablation Study
### Traditional Sparse Linear Models vs. Pre-Trained Contextual Sequence Networks

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-lightgrey)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.x-green)
![Course](https://img.shields.io/badge/IMC--601-Goa%20University-purple)

---

## Overview

Developed for **IMC-601: Introduction to Data Science** at Goa University. This project conducts a comprehensive ablation study comparing traditional sparse feature methods against modern dense sequence architectures for multi-class text classification, culminating in a proposed solution using pre-trained contextual sentence embeddings.

**Research Gap:** Existing models either sacrifice semantic depth (TF-IDF) or destroy spatial word order (Word2Vec averaging), while training sequence models (CNN/LSTM) from scratch on mid-sized corpora leads to severe overfitting and gradient collapse. This work proposes and validates a lightweight pre-trained transformer solution that resolves both limitations simultaneously.

---

## Dataset: 20 Newsgroups

| Split | Documents | Classes |
|:------|----------:|--------:|
| Train | 11,314 | 20 |
| Test | 7,532 | 20 |

A physical ETL pipeline was built to extract, store, and load the raw corpus from a local hierarchical directory structure (`data/full/train/<category>/doc_N.txt`), replicating a production-grade data engineering workflow.

---

## Repository Structure

```
.
├── src/
│   ├── step0_build_datasets_and_eda.py   # ETL pipeline + EDA visualizations
│   ├── step1_preprocessing.py            # Text cleaning, tokenization, label encoding
│   ├── step2_baseline_mnb.py             # Method 2: TF-IDF + Naive Bayes
│   ├── step3_baseline_svm.py             # Method 3: TF-IDF + Linear SVM
│   ├── step4_word2vec_fnn.py             # Method 4: Word2Vec + Feed-Forward NN
│   ├── step5_cnn_model.py                # Method 5: 1D Convolutional Neural Network
│   ├── step6_lstm_model.py               # Method 6: LSTM (W2V-initialized, fixed)
│   └── step7_sentence_transformers.py    # Method 7: Sentence Transformers + SVM (PROPOSED)
├── results/
│   └── eda/                              # Generated EDA plots (after running step0)
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build local dataset and generate EDA plots (run once)
cd src && python step0_build_datasets_and_eda.py

# 4. Run methods in order
python step2_baseline_mnb.py
python step3_baseline_svm.py
python step4_word2vec_fnn.py
python step5_cnn_model.py
python step6_lstm_model.py
python step7_sentence_transformers.py    # Proposed solution
```

---

## Experimental Methodology

| # | Method | Feature Representation | Classifier |
|:--|:-------|:----------------------|:-----------|
| 2 | Baseline | TF-IDF (sparse, 10k features) | Multinomial Naive Bayes |
| 3 | Traditional | TF-IDF (sparse, 10k features) | Linear SVM |
| 4 | Dense Averaging | Word2Vec Skip-gram (100d, mean-pooled) | Feed-Forward NN |
| 5 | Local Sequence | Learned Embeddings (100d) | 1D CNN (kernel=5) |
| 6 | Global Sequence | Word2Vec Init Embeddings (frozen, 100d) | LSTM (max_len=200) |
| **7** | **Proposed** | **Sentence Transformers, all-MiniLM-L6-v2 (384d)** | **Linear SVM** |

---

## Results

| Model | Accuracy | Macro F1-Score |
|:------|:--------:|:--------------:|
| **Method 7: Sentence Transformers + SVM (Proposed)** | **66.53%** | **65.03%** |
| Method 2: TF-IDF + MNB (Baseline) | 66.28% | 63.42% |
| Method 3: TF-IDF + SVM | 65.80% | 64.70% |
| Method 4: Word2Vec + FNN | 53.61% | 49.37% |
| Method 5: 1D CNN | 39.87% | 34.17% |
| Method 6: LSTM (Fixed, W2V Initialized) | 6.84% | 2.64% |

**Method 7 achieves the highest Macro F1-Score (65.03%) of the entire study.**

---

## Key Findings

**Why TF-IDF outperforms scratch-trained deep learning:** Sparse keyword matrices create highly linearly separable boundaries. CNN memorized training data (>91% train acc) but collapsed to 34% on test. LSTM plateaued below 8% training accuracy — data starvation, not architecture.

**Why averaging destroys performance (Method 4):** "Dog bites man" and "man bites dog" produce identical document vectors after mean-pooling. Word order is completely discarded.

**Why the proposed solution works (Method 7):** `all-MiniLM-L6-v2` was pre-trained on 1B+ sentence pairs, encoding semantic meaning, word order, and syntactic structure into 384d vectors — without any GPU fine-tuning on the target corpus.

---

## Dependencies

```
scikit-learn
nltk
gensim
tensorflow
sentence-transformers
numpy
pandas
matplotlib
seaborn
```

---

*Author: Rudresh Achari | IMC-601: Introduction to Data Science | Goa University*
