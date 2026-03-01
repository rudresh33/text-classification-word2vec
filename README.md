# Text Classification Ablation Study

**Traditional Sparse Linear Models vs. Pre-Trained Contextual Sequence Networks**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.x-4CAF50?style=flat-square)](https://sbert.net)
[![Course](https://img.shields.io/badge/IMC--601-Goa%20University-6A0DAD?style=flat-square)](https://unigoa.ac.in)

Developed for **IMC-601: Introduction to Data Science** at Goa University. This project conducts a controlled ablation study across six text classification methods, progressing from traditional sparse representations to a proposed pre-trained contextual embedding pipeline. Each method is designed to isolate the contribution of a specific representational or architectural choice, enabling causal attribution of performance differences.

---

## Research Motivation

Three representational failure modes are identified in the existing literature, each motivating a step in the ablation:

| Failure Mode | Root Cause | Methods Affected |
|---|---|---|
| Semantic blindness | TF-IDF assigns orthogonal dimensions to all vocabulary items regardless of meaning | M2, M3 |
| Word-order destruction | Document-level mean pooling discards positional and syntactic structure entirely | M4 |
| Data starvation | CNN and LSTM architectures require volumes of labelled data not available in this corpus | M5, M6 |

**Proposed solution:** `all-MiniLM-L6-v2` sentence embeddings applied in inference mode, combined with a Linear SVM. This configuration resolves all three failure modes simultaneously without GPU resources or task-specific fine-tuning.

---

## Dataset: 20 Newsgroups

The standard benchmark for multi-class text classification research, distributed across 20 topical categories spanning computing, science, politics, religion, and recreation.

| Partition | Documents | Share |
|---|---:|---:|
| Training | 11,314 | 60.1% |
| Test | 7,532 | 39.9% |
| **Total** | **18,846** | **100%** |
| Distinct Categories | 20 | Multi-class |

A local ETL pipeline was built to extract, store, and reload the corpus from a hierarchical directory structure (`data/full/train/<category>/doc_N.txt`), replicating a production-grade data engineering workflow. Metadata (headers, footers, quoted replies) was stripped prior to modelling to prevent target leakage.

---

## Repository Structure

```
.
├── src/
│   ├── step0_build_datasets_and_eda.py     # ETL pipeline + EDA visualisations
│   ├── step1_preprocessing.py              # Lowercasing, punctuation removal, tokenisation, stopword filtering
│   ├── step2_baseline_mnb.py               # M2: TF-IDF + Multinomial Naive Bayes
│   ├── step3_baseline_svm.py               # M3: TF-IDF + Linear SVM
│   ├── step4_word2vec_fnn.py               # M4: Word2Vec (Skip-gram) + Feed-Forward NN
│   ├── step5_cnn_model.py                  # M5: 1D Convolutional Neural Network
│   ├── step6_lstm_model.py                 # M6: LSTM with Word2Vec initialisation (frozen)
│   ├── step7_sentence_transformers.py      # M7: Sentence Transformers + SVM (PROPOSED)
│   └── step8_visualize_results.py          # Bar chart + confusion matrix generation
├── results/
│   ├── eda/
│   │   ├── class_distribution.png          # Documents per category (train vs. test)
│   │   └── word_count_distribution.png     # Document length histogram (log scale)
│   ├── model_comparison.png                # Ablation bar chart: Accuracy + Macro F1
│   └── method7_confusion_matrix.png        # Confusion matrix for the proposed solution
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the local dataset and generate EDA plots (run once)
cd src
python step0_build_datasets_and_eda.py

# 4. Run ablation methods in order
python step2_baseline_mnb.py
python step3_baseline_svm.py
python step4_word2vec_fnn.py
python step5_cnn_model.py
python step6_lstm_model.py
python step7_sentence_transformers.py    # Proposed solution

# 5. Generate performance comparison plots
python step8_visualize_results.py
```

> **Note:** `step1_preprocessing.py` is called internally by each modelling script. Running it standalone applies and saves the cleaned token lists for downstream use.

---

## Preprocessing Pipeline

All six methods receive identical preprocessed input. Performance differences are therefore attributable exclusively to representation and modelling choices.

| Step | Operation | Rationale |
|---|---|---|
| 1 | Lowercase conversion | Eliminates surface-form variation (`Computer` and `computer` unified) |
| 2 | Punctuation removal | Reduces typographic noise with no semantic cost for topic classification |
| 3 | Whitespace tokenisation | Produces the canonical token list consumed by all downstream methods |
| 4 | Stopword removal | Discards high-frequency function words (NLTK English corpus); retains content-bearing vocabulary |

---

## Ablation Methods

| # | Method | Representation | Classifier | Key Hyperparameters |
|---|---|---|---|---|
| M2 | TF-IDF + MNB | Sparse bag-of-words (10k features) | Multinomial Naive Bayes | `min_df=2`, `max_df=0.95`, `alpha=1.0` |
| M3 | TF-IDF + SVM | Sparse bag-of-words (10k features) | LinearSVC | `dual=False`, `random_state=42` |
| M4 | Word2Vec + FNN | Dense mean-pooled (100d) | Feed-Forward NN | `vector_size=100`, `window=5`, `128-64` neurons, `dropout=0.5` |
| M5 | 1D CNN | Learned embeddings (100d, `vocab=20k`) | Conv1D + GlobalMaxPool | `128 filters`, `kernel=5`, `max_len=500` |
| M6 | LSTM | Word2Vec init (frozen, 100d) | LSTM | `128 units`, `dropout=0.2`, `max_len=200` |
| **M7** | **Sentence Transformers + SVM** | **Contextual embeddings (384d)** | **LinearSVC** | **`all-MiniLM-L6-v2`, 22M params, `max_iter=2000`** |

> M1 is reserved for a random baseline. The ablation begins at M2.

---

## Results

| Method | Architecture | Accuracy | Macro F1 |
|---|---|---:|---:|
| **M7** | **Sentence Transformers + SVM (Proposed)** | **66.53%** | **65.03%** |
| M2 | TF-IDF + Multinomial Naive Bayes (Baseline) | 66.28% | 63.42% |
| M3 | TF-IDF + Linear SVM | 65.80% | 64.70% |
| M4 | Word2Vec + Feed-Forward NN | 53.61% | 49.37% |
| M5 | 1D CNN | 39.87% | 34.17% |
| M6 | LSTM (Word2Vec init, frozen) | 6.84% | 2.64% |

Primary evaluation metric: **Macro-Averaged F1-Score**. Macro averaging computes F1 independently per class and averages with equal weight, providing an unbiased measure under the near-uniform class distribution confirmed in EDA.

---

## Interpretation of Results

**M2 and M3 (TF-IDF):** Performance ceiling near 65% F1. TF-IDF vectors are highly linearly separable, which both MNB and LinearSVC exploit effectively. The ceiling is structural: terms with equivalent meaning but different surface forms occupy orthogonal dimensions and cannot be recognised as proximate.

**M4 (Word2Vec + FNN):** F1 drops to 49.37%, below the sparse baseline. Mean pooling over all word vectors produces a document representation with no positional information. The sentence "the defendant was not found guilty" and its semantic inverse produce nearly identical document vectors under this scheme.

**M5 (1D CNN):** Training accuracy exceeded 91% while test F1 reached only 34.17%. This severe disparity is unambiguous overfitting. Convolutional filters require large labelled volumes to learn discriminative patterns from random initialisation; approximately 11,000 training documents is insufficient.

**M6 (LSTM):** Training accuracy remained below 8% at early stopping (epoch 4). The failure is not architectural; it is data starvation. LSTM cells contain a large number of recurrently connected learnable parameters governing forget, input, and output gates. With fewer than 600 examples per class, the optimiser cannot find a reliable gradient signal.

**M7 (Proposed):** 65.03% Macro F1, the highest result in the study. `all-MiniLM-L6-v2` was pre-trained on over one billion sentence pairs using a contrastive Siamese objective. Its 384-dimensional output vectors encode semantic meaning, word order, and syntactic structure simultaneously. Because the encoder operates in inference mode with no task-specific fine-tuning, the pipeline runs entirely on CPU. The LinearSVC then finds optimal decision boundaries in a pre-structured, semantically rich embedding space.

---

## Key Findings

1. Sparse TF-IDF representations yield a robust baseline (~65% F1) bounded by semantic insensitivity to vocabulary relationships.
2. Document-level mean pooling of word vectors consistently degrades classification performance relative to sparse baselines.
3. Deep learning architectures trained from random initialisation require data volumes not available in this corpus; neither CNN nor LSTM generalises effectively.
4. Pre-trained contextual sentence embeddings overcome all three failure modes identified in the problem statement.
5. The proposed method achieves the highest Macro F1-Score (65.03%) without GPU resources or task-specific fine-tuning.

---

## Future Scope

- **End-to-end fine-tuning of BERT or RoBERTa** is projected to yield Macro F1-Scores in excess of 80% based on published benchmarks, at higher computational cost.
- **LSTM re-evaluation on larger corpora** would isolate architectural constraints from data-starvation effects, which this study cannot distinguish.
- **Preprocessing adjustment for M7**: Stopword removal prior to sentence encoding is suboptimal. Pre-trained transformers achieve best performance on unmodified natural prose; the preprocessing pipeline should be conditionally bypassed for sentence embedding methods.
- **Cross-corpus generalisation experiments** would assess the transferability of the proposed pipeline to other multi-class text classification benchmarks.
- **TF-IDF + contextual embedding ensemble** may provide marginal performance gains over either representation class in isolation.

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

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

**Author:** Rudresh Achari | Roll No. 2330
**Course:** IMC-601: Introduction to Data Science
**Institution:** MSc Integrated Data Science, Year III, Semester II | Goa Business School - Goa University
