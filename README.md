# Text Classification Ablation Study: Dense Embeddings vs. Sparse Representations

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-lightgrey)

## 📌 Project Overview
Developed for the **IMC-601: Introduction to Data Science** course at Goa University. This repository contains a comprehensive ablation study comparing traditional sparse linear models against modern dense sequence deep learning architectures for text classification. 

## 📊 Dataset: 20 Newsgroups
A physical local pipeline was built to extract, clean, and load the raw text corpus:
* **Training Documents:** 11,314
* **Testing Documents:** 7,532
* **Classes:** 20

## 🛠️ Experimental Methodology
The study transitions from context-blind frequency counting to spatial sequence learning:
1. **Method 2 & 3 (Traditional):** TF-IDF Vectorization paired with Multinomial Naive Bayes and Linear Support Vector Machines.
2. **Method 4 (Dense Vector Averaging):** Custom Word2Vec embeddings (Skip-gram, 100d) averaged into document vectors and fed to a Shallow Feed-Forward Neural Network (FNN).
3. **Method 5 (Local Sequence):** 1D Convolutional Neural Network (CNN) with a sliding n-gram kernel.
4. **Method 6 (Global Sequence):** Long Short-Term Memory (LSTM) recurrent network.

## 🚀 Results

| Model Architecture | Accuracy | Macro-Averaged F1-Score |
| :--- | :--- | :--- |
| **Method 3: TF-IDF + SVM** | **65.80%** | **64.70%** |
| Method 2: TF-IDF + MNB (Baseline) | 66.28% | 63.42% |
| Method 4: Word2Vec + FNN | 53.61% | 49.37% |
| Method 5: 1D CNN | 39.87% | 34.17% |
| Method 6: LSTM | 5.31% | 0.55% |

## 💡 Discussion & Conclusion
The traditional **TF-IDF + SVM pipeline vastly outperformed all deep learning methods**. 

While dense embeddings successfully capture word-level semantics, applying Document Vector Averaging (Method 4) destroyed crucial spatial word order, limiting performance. Conversely, attempting to preserve sequences using deep learning (CNNs and LSTMs) from scratch on a mid-sized corpus led to severe overfitting and gradient collapse. The CNN memorized the training data but failed to generalize, while the LSTM could not maintain context across 500-word sequences without an Attention mechanism.

This study confirms that for document-level text classification, sparse feature spaces provide stronger, more robust linear separability than improperly initialized sequence networks.

---
*Author: Rudresh Achari*
