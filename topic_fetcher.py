import wikipedia
def get_topic(topic: str, word_count: int):
    """ Returns a specified topic from wikipedia of `word_count` words """
    page = wikipedia.page(topic)
    topic = page.content[:word_count]
    
    return topic