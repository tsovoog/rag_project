from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    """
    nomic-embed-text загвар ашиглана.
    Энэ нь олон хэлийг дэмждэг, хурдан, локал embedding загвар.
    Суулгах: ollama pull nomic-embed-text
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
