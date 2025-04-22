# PREPARED BY: CARADO & TACUEL

# QUESTION 1
# 1. (20 points) Using Wikipedia as the corpus, 
# obtain 5 different topics that will serve as your 
# documents, and create a term-document matrix. 
# You can use the shared code on GitHub as a reference.

# a. Term-document matrix using raw frequency.
# b. Term-document matrix using TF-IDF weights.

from utils import get_topic, preprocess

CHARACTERS_COUNT = 1000

# Getting the topics
napoleon = get_topic('Napoleon', 100)
donald = get_topic('Donald_Duck', CHARACTERS_COUNT)
mickey = get_topic('Mickey_Mouse', CHARACTERS_COUNT)
minnie = get_topic('Minnie_Mouse', CHARACTERS_COUNT)
tweety = get_topic('Tweety', CHARACTERS_COUNT)
