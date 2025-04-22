# QUESTION 5
# (20 points) What are the differences of using Word2Vec compared to the tf-idf in terms of:
# - Vector Space?
# - Vector Size?

from utils import get_topic, preprocess
from compute import compute_tf, compute_idf, compute_tfidf, cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import nltk
nltk.download('punkt')

"""
What this file will do is make TF-IDF model and Word2Vec model and compare them.
"""

CHARACTERS_COUNT = 1000

# Getting the topics
napoleon = get_topic('Napoleon', CHARACTERS_COUNT)
oppenheimer = get_topic('Robert J. Oppenheimer', CHARACTERS_COUNT)
alexander = get_topic('Alexander the Great', CHARACTERS_COUNT)
marcus = get_topic('Marcus Aurelius', CHARACTERS_COUNT)
winston = get_topic('Winston Churchill', CHARACTERS_COUNT)

documents = [napoleon, oppenheimer, alexander, marcus, winston]

tokenized_docs = [preprocess(doc) for doc in documents]

# === TF-IDF === (Copypaste Q3)

# Create a set of unique words (vocabulary)
vocabulary = set(word for doc in tokenized_docs for word in doc)

# Compute TF for each document
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

# Compute IDF
idf = compute_idf(tokenized_docs, vocabulary)

tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

# Print vector size and sparsity
print("==== TF-IDF ====")
print("TF-IDF vector space size:", len(vocabulary))
non_zero_counts = [sum(1 for val in vec.values() if val != 0) for vec in tfidf_vectors]
sparsity = 100 * (1 - np.mean(non_zero_counts) / len(vocabulary))
print(f"TF-IDF avg sparsity: {sparsity:.2f}%")

# Print cosine similarity matrix
print("TF-IDF cosine similarity matrix:")
for i in range(len(tfidf_vectors)):
    row = []
    for j in range(len(tfidf_vectors)):
        sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocabulary)
        row.append(f"{sim:.2f}")
    print("  ".join(row))

# === Word2Vec ===

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_docs)

# Convert document to vector (average of word vectors)
def document_vector(doc, model):
    valid_words = [word for word in doc if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)

w2v_vectors = np.array([document_vector(doc, w2v_model) for doc in tokenized_docs])

# Print vector size and sparsity
print("\n==== Word2Vec ====")
print("Word2Vec vector size:", w2v_vectors.shape[1])
print("Word2Vec sparsity: 0.00% (dense vectors)")

# Cosine similarity matrix for Word2Vec
print("Word2Vec cosine similarity matrix:")
cos_sim_matrix = np.round(np.dot(w2v_vectors, w2v_vectors.T) / (
    np.linalg.norm(w2v_vectors, axis=1)[:, None] * np.linalg.norm(w2v_vectors, axis=1)[None, :]), 2)
print(cos_sim_matrix)