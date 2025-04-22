# CCS249 Unit 5 Assignment

**Authors:** John Manuel Carado & Allan Andrews Tacuel  
**Date:** April 22, 2025  
**Course:** BSCS 3-A

This repository contains code and documentation for the Unit 5 assignment in our Natural Language Processing course. The task involves text processing, vectorization, similarity computation, and classification using traditional and modern NLP techniques.

---

## 🧠 Problem Set Summary

### 1. Term-Document Matrix

**Corpus:** Five Wikipedia articles on notable historical figures.

#### a. Raw Frequency
We generated term-document matrices using raw frequency. Below are some sample frequencies:

- **[Napoleon](https://en.wikipedia.org/wiki/Napoleon):** `{'french': 0.08, 'napoleon': 0.03, 'revolution': 0.02, ...}`
- **[Oppenheimer](https://en.wikipedia.org/wiki/J._Robert_Oppenheimer):** `{'oppenheimer': 0.042, 'physics': 0.042, 'university': 0.031, ...}`
- **[Alexander the Great](https://en.wikipedia.org/wiki/Alexander_the_Great):** `{'bc': 0.040, 'alexander': 0.040, 'age': 0.030, ...}`
- **[Marcus Aurelius](https://en.wikipedia.org/wiki/Alexander_the_Great):** `{'marcus': 0.065, 'antoninus': 0.037, 'roman': 0.028, ...}`
- **[Winston Churchill](https://en.wikipedia.org/wiki/Winston_Churchill):** `{'war': 0.042, 'british': 0.031, '1900': 0.021, ...}`

#### b. TF-IDF Weights
Applying TF-IDF emphasized more contextually distinctive terms:

- **Napoleon:** `{'french': 0.129, 'napoleon': 0.048, 'revolution': 0.032, ...}`
- **Oppenheimer:** `{'oppenheimer': 0.068, 'physics': 0.068, 'nuclear': 0.034, ...}`
- **Alexander the Great:** `{'alexander': 0.065, 'macedon': 0.049, 'greek': 0.033, ...}`
- **Marcus Aurelius:** `{'marcus': 0.104, 'antoninus': 0.0596, '180': 0.0447, ...}`
- **Winston Churchill:** `{'british': 0.0502, 'war': 0.0381, '1955': 0.0335 ...}`

> View the submitted document for completed answers.

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

- **Cosine Similarity:** It measures how close two vectors are in space. Higher values = more similar meaning
- TF-IDF: low scores (~0.01), meaning it failed to capture meanings among words
- Word2Vec: high scores (up to 0.7), even with different wording
- **Negative cosine values** in Word2Vec indicate opposing semantic meanings

> **Conclusion:**  
> Word2Vec captures deeper and more nuanced semantic relationships between documents than TF-IDF.

---

## 📁 Repository Contents

- `Q1-2-3.py`: Answers the first three questions of this assignment.
- `Q4.py ... Q6.py`: Answers the next three questions of this assignment.
- `utils.py`: Includes multiple helper methods such as text preprocessing

> The answers can be found above.

---

## 💻 Setup

- Clone the repository by running this command: `git clone https://github.com/hydraadra112/CCS249-Unit5-Assignment-CARADO-TACUEL.git`
- Change directory by `cd .\CCS249-Unit5-Assignment-CARADO-TACUEL`
- Create a virtual environment via `python -m venv .venv`
- Activate the virtual environment by `.venv\Scripts\activate`
- Install dependencies by `pip install -r .\requirements.txt`
- And you're all set! You can run one Python file by `python Q1-2-3.py` to see an example output.

---

## ✍️ Authors

- **John Manuel Carado**  
- **Allan Andrews Tacuel**

---

## 📝 License

This project is for academic purposes only.

