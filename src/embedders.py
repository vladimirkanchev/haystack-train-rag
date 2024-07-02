"""Contain wrappers of embedder components of tne rag pipeline."""

from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder


def setup_embedder(model_name: str) -> FastembedTextEmbedder:
    """Transform a string into a vector."""
    return FastembedTextEmbedder(model=model_name)
