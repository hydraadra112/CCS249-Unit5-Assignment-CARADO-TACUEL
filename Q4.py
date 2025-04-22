# QUESTION 4
# (30 points) Using the same dataset used above, use the word2vec package to
# create a classifier for dense vectors.
# a. Use Logistic Regression, with the appropriate configuration for the model and dataset.

from utils import get_topic, preprocess
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
nltk.download('punkt')

"""
What this file will do is try to classify the documents into classes.
The documents are pertaining to powerful historical figures.
We have classified them into 3 classes: Scientist, Military Leader, and Political Leader.
We try to use Logreg to predict them after we have word2vec'd them.
"""

CHARACTERS_COUNT = 1000
napoleon = get_topic('Napoleon', CHARACTERS_COUNT) # Document 1
oppenheimer = get_topic('J. Robert Oppenheimer', CHARACTERS_COUNT) # Document 2
alexander = get_topic('Alexander_the_Great', CHARACTERS_COUNT) # Document 3
marcus = get_topic('Marcus_Aurelius', CHARACTERS_COUNT) # Document 4
winston = get_topic('Winston_Churchill', CHARACTERS_COUNT) # Document 5

documents = [napoleon, oppenheimer, alexander, marcus, winston]

tokenized_docs = [preprocess(doc) for doc in documents]

model = Word2Vec(sentences=tokenized_docs)

# Function to convert each doc to a vector (by averaging word vectors)
def document_vector(doc, model):
    valid_words = [word for word in doc if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)

# Create document vectors
X = np.array([document_vector(doc, model) for doc in tokenized_docs])

# Labels
# 0 = Scientist
# 1 = Military Leader
# 2 = Political Leader

y = [1, 0, 1, 2, 2]

# Train logreg model
clf = LogisticRegression()
clf.fit(X, y)

# Predict on same data
predictions = clf.predict(X)

print("Predicted classes for the documents:")
for i, label in enumerate(predictions):
    print(f"Document {i+1}: Class {label}")