# QUESTION 6
# (10 points) How do we evaluate the performance of Semantic Models (i.e TF-IDF and Word2Vec)?

from utils import get_topic, preprocess
from compute import compute_tf, compute_idf, compute_tfidf, cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import nltk
nltk.download('punkt')

CHARACTERS_COUNT = 1000

# Topics
napoleon = get_topic('Napoleon', CHARACTERS_COUNT)
oppenheimer = get_topic('Robert J. Oppenheimer', CHARACTERS_COUNT)
alexander = get_topic('Alexander the Great', CHARACTERS_COUNT)
marcus = get_topic('Marcus Aurelius', CHARACTERS_COUNT)
winston = get_topic('Winston Churchill', CHARACTERS_COUNT)

documents = [napoleon, oppenheimer, alexander, marcus, winston]

tokenized_docs = [preprocess(doc) for doc in documents]

# === TF-IDF Cosine Evaluation ===
vocab = set(word for doc in tokenized_docs for word in doc)
tf_vectors = [compute_tf(doc, vocab) for doc in tokenized_docs]
idf = compute_idf(tokenized_docs, vocab)
tfidf_vectors = [compute_tfidf(tf, idf, vocab) for tf in tf_vectors]

print("==== TF-IDF Cosine Similarity ====")
for i in range(len(tfidf_vectors)):
    row = []
    for j in range(len(tfidf_vectors)):
        sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocab)
        row.append(f"{sim:.2f}")
    print("  ".join(row))

# === Word2Vec Cosine Evaluation ===
w2v_model = Word2Vec(sentences=tokenized_docs)

def document_vector(doc, model):
    valid_words = [w for w in doc if w in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in valid_words], axis=0)

w2v_vectors = np.array([document_vector(doc, w2v_model) for doc in tokenized_docs])

print("\n==== Word2Vec Cosine Similarity ====")
cos_sim_matrix = np.round(np.dot(w2v_vectors, w2v_vectors.T) / (
    np.linalg.norm(w2v_vectors, axis=1)[:, None] * np.linalg.norm(w2v_vectors, axis=1)[None, :]), 2)
print(cos_sim_matrix)