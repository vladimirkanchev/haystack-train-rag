"""Contain wrappers of retriever components of tne rag pipeline."""
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

import box
import yaml

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_single_retriever(doc_store):
    """Build embedding or sparse(bm25)-based retreiver."""
    retriever = None
    if cfg.TYPE_RETRIEVAL == 'dense':
        retriever = InMemoryEmbeddingRetriever(doc_store)
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        retriever = InMemoryBM25Retriever(document_store=doc_store)
    return retriever


def setup_hyrbrid_retriever(doc_store):
    """Build embedding and sparse(bm25)-based retreiver."""
    dense_retriever = InMemoryEmbeddingRetriever(doc_store)
    sparse_retriever = InMemoryBM25Retriever(doc_store)
    return dense_retriever, sparse_retriever
