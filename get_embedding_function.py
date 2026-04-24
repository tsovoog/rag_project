from langchain_ollama import OllamaEmbeddings
from mongolian_utils import normalize_mongolian


class MongolianEmbeddings:
    def __init__(self):
        self.base = OllamaEmbeddings(model="nomic-embed-text")

    def embed_documents(self, texts: list) -> list:
        normalized = [normalize_mongolian(t) for t in texts]
        return self.base.embed_documents(normalized)

    def embed_query(self, text: str) -> list:
        normalized = normalize_mongolian(text)
        return self.base.embed_query(normalized)


def get_embedding_function():
    return MongolianEmbeddings()