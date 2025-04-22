# CCS249 Unit 5 Assignment

**Authors:** John Manuel Carado & Allan Andrews Tacuel  
**Date:** April 22, 2025  
**Course:** BSCS 3-A

This repository contains code and documentation for the Unit 5 assignment in our Computational Semantics course. The task involves text processing, vectorization, similarity computation, and classification using traditional and modern NLP techniques.

---

## 🧠 Problem Set Summary

### 1. Term-Document Matrix

**Corpus:** Five Wikipedia articles on notable historical figures.

#### a. Raw Frequency
We generated term-document matrices using raw frequency. Below are some sample frequencies:

- **Napoleon:** `{'french': 0.08, 'napoleon': 0.03, 'revolution': 0.02, ...}`
- **Oppenheimer:** `{'oppenheimer': 0.042, 'physics': 0.042, 'university': 0.031, ...}`
- **Alexander the Great:** `{'bc': 0.040, 'alexander': 0.040, 'age': 0.030, ...}`
- **Marcus Aurelius:** `{'marcus': 0.065, 'antoninus': 0.037, 'roman': 0.028, ...}`
- **Winston Churchill:** `{'war': 0.042, 'british': 0.031, '1900': 0.021, ...}`

#### b. TF-IDF Weights
Applying TF-IDF emphasized more contextually distinctive terms:

- **Napoleon:** `{'french': 0.129, 'napoleon': 0.048, 'revolution': 0.032, ...}`
- **Oppenheimer:** `{'oppenheimer': 0.068, 'physics': 0.068, 'nuclear': 0.034, ...}`
- **Alexander the Great:** `{'alexander': 0.065, 'macedon': 0.049, 'greek': 0.033, ...}`

### 2. Comparison: TF-IDF vs Raw Frequency

- **Raw frequency** prioritizes common terms within the document.
- **TF-IDF** reduces importance of frequent corpus-wide terms, highlighting unique document-specific ones.
- TF-IDF provides better **semantic discrimination**.

---

### 3. Cosine Similarity of Documents

Cosine similarity was calculated pairwise across documents:

| Document Pair                     | Similarity |
|----------------------------------|------------|
| Napoleon - Churchill             | **0.0162** |
| Oppenheimer - Churchill          | **0.0274** |
| Alexander - Marcus Aurelius      | **0.0180** |

**Most similar:**  
- **J. Robert Oppenheimer & Winston Churchill (0.0274)**  
- Likely due to overlapping WWII-related vocabulary and context.

---

### 4. Word2Vec + Logistic Regression Classifier

**Steps:**
1. Preprocessing with `preprocess()` from `utils.py`
2. Word embedding via `gensim.models.Word2Vec` (default settings)
3. Document vectorization by averaging word vectors
4. Labels defined by thematic classes (e.g., Scientist vs Military Leader)
5. Classification via **Logistic Regression**

**Results:**  
- **4 out of 5 documents correctly predicted**
- Demonstrates the effectiveness of Word2Vec embeddings in semantic classification tasks.

---

### 5. TF-IDF vs Word2Vec

| Feature             | TF-IDF                            | Word2Vec                         |
|---------------------|-----------------------------------|----------------------------------|
| Vector Space        | Sparse (357 dims)                 | Dense (100 dims)                |
| Vector Size         | Based on vocab size               | Fixed size (hyperparam)         |
| Semantic Depth      | Low (based on frequency)          | High (based on context/meaning) |

---

### 6. Evaluation of Semantic Models

- **Cosine Similarity** used for performance evaluation
- TF-IDF: low scores (~0.01) unless same words used
- Word2Vec: high scores (up to 0.7), even with different wording
- **Negative cosine values** in Word2Vec indicate opposing semantic meanings

> **Conclusion:**  
> Word2Vec captures deeper and more nuanced semantic relationships between documents than TF-IDF.

---

## 📁 Repository Contents

- `tfidf_raw.py`: Generates raw frequency term-document matrix
- `tfidf_model.py`: Computes TF-IDF weighted vectors
- `cosine_similarity.py`: Performs pairwise document similarity computation
- `word2vec_classifier.py`: Trains and tests logistic regression using word embeddings
- `utils.py`: Preprocessing utility for text cleaning and normalization

---

## 🔗 Repository

[GitHub Repository: CCS249-Unit5-Assignment-CARADO-TACUEL](https://github.com/hydraadra112/CCS249-Unit5-Assignment-CARADO-TACUEL)

---

## ✍️ Authors

- **John Manuel Carado**  
- **Allan Andrews Tacuel**

---

## 📝 License

This project is for academic purposes only.

