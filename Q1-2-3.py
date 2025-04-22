# PREPARED BY: CARADO & TACUEL

# QUESTION 1
# 1. (20 points) Using Wikipedia as the corpus, 
# obtain 5 different topics that will serve as your 
# documents, and create a term-document matrix. 
# You can use the shared code on GitHub as a reference.

# a. Term-document matrix using raw frequency.
# b. Term-document matrix using TF-IDF weights.

from utils import get_topic, preprocess, sort_dict_by_value_desc
from compute import compute_tf, compute_idf, compute_tfidf, cosine_similarity

CHARACTERS_COUNT = 1000

# Getting the topics
napoleon = get_topic('Napoleon', CHARACTERS_COUNT)
oppenheimer = get_topic('J.R. Oppenheimer', CHARACTERS_COUNT)
alexander = get_topic('Alexander_the_Great', CHARACTERS_COUNT)
marcus = get_topic('Marcus_Aurelius', CHARACTERS_COUNT)
winston = get_topic('Winston_Churchill', CHARACTERS_COUNT)

documents = [napoleon, oppenheimer, alexander, marcus, winston]

tokenized_docs = [preprocess(doc) for doc in documents]

# Create a set of unique words (vocabulary)
vocabulary = set(word for doc in tokenized_docs for word in doc)

# Compute the term frequency for each document
tf_vectors =  [compute_tf(doc, vocabulary) for doc in tokenized_docs ]

print("\nTerm Frequency Vectors:")
for i, tf_vector in enumerate(tf_vectors):
    print(f"Term Frequency Vector of Document {i+1}: {sort_dict_by_value_desc(tf_vector)}\n\n")

# Compute the Inverse Document Frequency (IDF)
idf = compute_idf(tokenized_docs, vocabulary)
print("\n\n\nInverse Document Frequency:")
for term, idf_value in idf.items():
    print(f"{term}: {idf_value}")

tfidf_vectors = [ compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

print("\n\n\nTF-IDF Vectors:") 
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"TF-IDF Vector of Document {i+1}: {sort_dict_by_value_desc(tfidf_vector)}\n\n")

# Compute the Cosine Similarity between the first two documents
similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1], vocabulary)
print("\nCosine Similarity between Document 1 and Document 2:")
print(similarity)