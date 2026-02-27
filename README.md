# Text Classification: Dense Word Embeddings vs. Sparse Representations

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-lightgrey)

## 📌 Project Overview
This project was developed for the **IMC-601: Introduction to Data Science** course.  
It explores the evolution of text classification by comparing a traditional sparse feature approach against a modern dense embedding pipeline.

The experiment aims to address the performance limitations caused by sparse feature vectors and unrealistic feature independence assumptions in traditional models.

## 📊 Dataset
We utilized the standard **20 Newsgroups Dataset** (available via scikit-learn and the UCI Machine Learning Repository).

- **Total Categories:** 20  
- **Training Documents:** 11,314  
- **Testing Documents:** 7,532  

## 🛠️ Methodology

### 1. Baseline Method (TF-IDF + Multinomial Naive Bayes)
A traditional approach relying on keyword frequencies.

**Feature Extraction:** TF-IDF Vectorizer  
- `max_features`: 10,000  
- `min_df`: 2  
- `max_df`: 0.95  

**Classifier:** Multinomial Naive Bayes  
- Laplace smoothing `alpha = 1.0`

---

### 2. Proposed Method (Word2Vec + Feed-Forward Neural Network)
A semantic-aware feature space mapped to a non-linear network.

**Feature Extraction:** Custom Word2Vec model  
- Dimension: 100  
- Window size: 5  
- Min frequency: 2  
- Epochs: 10  

Document vectors were generated via element-wise averaging of token vectors.

**Classifier:** Shallow Feed-Forward Neural Network (FNN)  
- Input Layer: 100 neurons  
- Hidden Layer 1: 128 neurons (ReLU)  
- Hidden Layer 2: 64 neurons (ReLU)  
- Dropout Rate: 0.5  
- Output Layer: 20 neurons (Softmax)  
- Optimizer: Adam (`learning_rate = 0.001`)

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/rudresh33/text-classification-word2vec.git
cd text-classification-word2vec
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Scripts

Run the modules in the following order to replicate the experiment:

```bash
# 1. Text preprocessing
python src/step1_preprocessing.py

# 2. Run the Baseline Model
python src/step2_baseline_tfidf.py

# 3. Train Word2Vec Embeddings
python src/step4_custom_word2vec.py

# 4. Train the Neural Network
python src/step6_neural_network.py
```

---

## 📈 Results

| Model Architecture | Accuracy | Macro-Averaged F1-Score |
|--------------------|----------|--------------------------|
| **Proposed Method (Word2Vec + FNN)** | **55.38%** | **52.07%** |
| **Baseline Method (TF-IDF + MNB)** | **66.28%** | **63.42%** |

---

## 💡 Discussion & Conclusion

Contrary to the initial hypothesis, the classic Multinomial Naive Bayes baseline outperformed the proposed non-linear FNN model.

While dense Word2Vec embeddings captured semantic similarities missed by sparse representations, the use of document vector averaging proved limiting.

Averaging vectors removes:

- Word order  
- Grammatical structure  
- Contextual nuance  

For example:

> "dog bites man"  
> "man bites dog"

Both produce identical averaged vectors despite opposite meanings.

Additionally, out-of-vocabulary words receive no representation.

This confirms that for long-form text classification, sparse keyword-frequency matrices like TF-IDF can provide stronger discriminative signals than aggressively averaged dense vectors used with shallow neural networks.

---

## 🔮 Future Work

Future improvements will explore sequence-aware models such as:

- Bidirectional LSTMs  
- Transformer-based architectures (BERT, RoBERTa)

These models preserve grammatical and semantic flow while leveraging dense embeddings more effectively.

---

**Author:** Rudresh Achari

