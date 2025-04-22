import wikipedia
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def get_topic(topic: str, character_count: int):
    """ Returns a specified topic from wikipedia of `character_count` words """
    page = wikipedia.page(topic)
    topic = page.content[:character_count]
    
    return topic

def preprocess(corpus: str) -> list[str]:
    """
    Performs full preprocessing on input text:
    - Lowercasing
    - Punctuation removal
    - Tokenization
    - Stopword removal
    Returns a list of cleaned tokens.
    """
    # Lowercase and remove punctuation
    lowered = corpus.lower()
    no_punct = lowered.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(no_punct)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    
    return cleaned_tokens
