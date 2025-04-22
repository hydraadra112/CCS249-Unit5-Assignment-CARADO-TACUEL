import wikipedia
def get_topic(topic: str, character_count: int):
    """ Returns a specified topic from wikipedia of `character_count` words """
    page = wikipedia.page(topic)
    topic = page.content[:character_count]
    
    return topic