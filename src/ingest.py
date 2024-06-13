"""Import files to build rag algorithm."""
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore

import box
from dotenv import load_dotenv, find_dotenv
import yaml

load_dotenv(find_dotenv())

# Import config vars
with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def load_data_no_preprocessing():
    """Load preprocessed dataset of seven wonders."""
    document_store = InMemoryDocumentStore()
    dataset = load_dataset(cfg.DATA_SET, split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"])
            for doc in dataset]

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=cfg.EMBEDDINGS)
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])
    return document_store


def main():
    """Start load data with/wo preprocessing algorithm."""
    load_data_no_preprocessing()


if __name__ == "__main__":
    main()
