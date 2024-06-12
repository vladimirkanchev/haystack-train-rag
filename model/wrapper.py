from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

import box
import sys
import yaml

from model.llm import setup_llm
from model.prompts import prompt_template
sys.path.append(".src")
from src import *


with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_prompt():
    return PromptBuilder(template = prompt_template)


def setup_embedder(model_name):
    return SentenceTransformersTextEmbedder(model = model_name)


def setup_retriever(doc_store):
    return InMemoryEmbeddingRetriever(doc_store)



def setup_rag_pipeline():
    document_store = InMemoryDocumentStore()

    prompt = setup_prompt()
    llm = setup_llm(cfg.LLM_MODEL)
    text_embedder = setup_embedder(cfg.EMBEDDINGS)
    retriever = setup_retriever(document_store)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt)
    rag_pipeline.add_component("llm", llm)

    # Now, connect the components to each other
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")


    return rag_pipeline