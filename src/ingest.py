"""Import files to build rag algorithm."""
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice

import box
from dotenv import load_dotenv, find_dotenv
import yaml


load_dotenv(find_dotenv())
# Import config vars
with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

document_store = InMemoryDocumentStore()
dataset = load_dataset(cfg.DATA_SET, split="train")
docs = [Document(content=doc["content"], meta=doc["meta"])
        for doc in dataset]

if cfg.TYPE_RETRIEVAL == 'dense':
    doc_embedder = SentenceTransformersDocumentEmbedder(
        device=ComponentDevice.from_str("cuda:0"),
        model=cfg.EMBEDDINGS)
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    FINAL_DOCS = docs_with_embeddings["documents"]
elif cfg.TYPE_RETRIEVAL == 'sparse':
    FINAL_DOCS = docs
elif cfg.TYPE_RETRIEVAL == 'hybrid':
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=cfg.EMBEDDINGS,
        device=ComponentDevice.from_str("cuda:0"))
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    FINAL_DOCS = docs_with_embeddings["documents"]
else:
    FINAL_DOCS = None

if FINAL_DOCS:
    document_store.write_documents(FINAL_DOCS)
