"""Import files to build rag algorithm."""
from datasets import load_dataset
from haystack import Document
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder 
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
    doc_embedder = FastembedDocumentEmbedder(
        device=ComponentDevice.from_str("cuda:0"),
        model=cfg.EMBEDDINGS)
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    final_docs = docs_with_embeddings["documents"]
elif cfg.TYPE_RETRIEVAL == 'sparse':
    final_docs = docs
elif cfg.TYPE_RETRIEVAL == 'hybrid':
    doc_embedder = FastembedDocumentEmbedder(
        model=cfg.EMBEDDINGS,
        device=ComponentDevice.from_str("cuda:0"))
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    final_docs = docs_with_embeddings["documents"]
else:
    final_docs = None

    if final_docs:
        document_store.write_documents(final_docs)

