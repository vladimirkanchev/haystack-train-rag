"""Import files to build rag algorithm."""
from pathlib import Path
import os
import sys

from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice

import box
from dotenv import load_dotenv, find_dotenv
import yaml

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

load_dotenv(find_dotenv())
# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def load_data_into_store():
    """Load and embed data into the in-memory document store."""
    document_store = InMemoryDocumentStore()
    dataset = load_dataset(cfg.DATA_SET, split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"])
            for doc in dataset]

    if cfg.TYPE_RETRIEVAL == 'dense':
        doc_embedder = SentenceTransformersDocumentEmbedder(
             model=cfg.EMBEDDINGS)
        doc_embedder.warm_up()

        docs_with_embeddings = doc_embedder.run(docs)
        final_docs = docs_with_embeddings["documents"]
        print(final_docs)
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        final_docs = docs
    elif cfg.TYPE_RETRIEVAL == 'hybrid':
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=cfg.EMBEDDINGS,
            device=ComponentDevice.from_str("cuda:0"))
        doc_embedder.warm_up()

        docs_with_embeddings = doc_embedder.run(docs)
        final_docs = docs_with_embeddings["documents"]
    else:
        final_docs = None

    if final_docs:
        document_store.write_documents(final_docs)

    return document_store
