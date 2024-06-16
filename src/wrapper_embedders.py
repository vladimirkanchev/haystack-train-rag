""" """
from haystack.components.embedders import SentenceTransformersTextEmbedder


def setup_embedder(model_name):
    """Transform a string into a vector."""
    return SentenceTransformersTextEmbedder(model=model_name)