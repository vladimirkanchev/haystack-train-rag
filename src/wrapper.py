"""Contain wrapper function of separater rag pipeline components."""
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


import box
import yaml

from llm import setup_single_llm
from prompts import PROMPT_TEMPLATE
from ingest import load_data_no_preprocessing

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_prompt():
    """Render a prompt template."""
    return PromptBuilder(template=PROMPT_TEMPLATE)


def setup_embedder(model_name):
    """Transform a string into a vector."""
    return SentenceTransformersTextEmbedder(model=model_name)


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


def setup_rag_dense_pipeline():
    """Build basic rag haystack pipeline."""
    doc_store = load_data_no_preprocessing()
    prompt = setup_prompt()
    llm = setup_single_llm(cfg.LLM_MODEL)
    text_embedder = setup_embedder(cfg.EMBEDDINGS)
    retriever = setup_single_retriever(doc_store)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt)
    rag_pipeline.add_component("llm", llm)

    # Now, connect the components to each other
    rag_pipeline.connect("text_embedder.embedding",
                         "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.draw(path=cfg.PIPELINE_PATH)

    return rag_pipeline


def setup_rag_sparse_pipeline():
    """Build basic rag haystack pipeline."""
    doc_store = load_data_no_preprocessing()
    prompt = setup_prompt()
    llm = setup_single_llm(cfg.LLM_MODEL)
    bm25_retriever = setup_single_retriever(doc_store)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", bm25_retriever)
    rag_pipeline.add_component("prompt_builder", prompt)
    rag_pipeline.add_component("llm", llm)

    # Now, connect the components to each other
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.draw(path=cfg.PIPELINE_PATH)

    return rag_pipeline


def setup_rag_hybrid_pipeline():
    """Build basic rag haystack pipeline."""
    doc_store = load_data_no_preprocessing()
    prompt = setup_prompt()
    llm = setup_single_llm(cfg.LLM_MODEL)
    text_embedder = setup_embedder(cfg.EMBEDDINGS)
    embedding_retriever, bm25_retriever = setup_hyrbrid_retriever(doc_store)

    document_joiner = DocumentJoiner()
    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component("text_embedder", text_embedder)
    hybrid_pipeline.add_component("embedding_retriever", embedding_retriever)
    hybrid_pipeline.add_component("bm25_retriever", bm25_retriever)
    hybrid_pipeline.add_component("document_joiner", document_joiner)
    hybrid_pipeline.add_component("ranker", ranker)
    hybrid_pipeline.add_component("prompt_builder", prompt)
    hybrid_pipeline.add_component("llm", llm)
    print(llm)

    # Now, connect the components to each other
    hybrid_pipeline.connect("text_embedder",
                            "embedding_retriever.query_embedding")
    hybrid_pipeline.connect("bm25_retriever", "document_joiner.documents")
    hybrid_pipeline.connect("embedding_retriever",
                            "document_joiner.documents")
    hybrid_pipeline.connect("document_joiner", "ranker")
    hybrid_pipeline.connect("ranker", "prompt_builder.documents")
    hybrid_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    hybrid_pipeline.draw(path=cfg.PIPELINE_PATH)

    return hybrid_pipeline
