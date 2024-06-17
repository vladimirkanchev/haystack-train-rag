"""Contain wrapper function of separater rag pipeline components."""
from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker

import box
import yaml

from .llm import setup_single_llm
from .ingest import load_data_no_preprocessing

from .wrapper_embedders import setup_embedder
from .wrapper_prompts import setup_prompt
from .wrapper_retrievers import setup_single_retriever
from .wrapper_retrievers import setup_hyrbrid_retriever

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


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
    hybrid_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
  

    # Now, connect the components to each other
    hybrid_pipeline.connect("text_embedder",
                            "embedding_retriever.query_embedding")
    hybrid_pipeline.connect("bm25_retriever", "document_joiner.documents")
    hybrid_pipeline.connect("embedding_retriever",
                            "document_joiner.documents")
    hybrid_pipeline.connect("document_joiner", "ranker")
    hybrid_pipeline.connect("ranker", "prompt_builder.documents")
    hybrid_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    hybrid_pipeline.connect("llm.replies", "answer_builder.replies")
    hybrid_pipeline.connect("llm.meta", "answer_builder.meta")
    hybrid_pipeline.connect("ranker", "answer_builder.documents")
    hybrid_pipeline.draw(path=cfg.PIPELINE_PATH)

    return hybrid_pipeline
