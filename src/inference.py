from haystack import Pipeline

import box
import yaml

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def run_pipeline(query: str, rag_pipeline: Pipeline) -> Pipeline:
    """Run rag/no rag pipeline with predifined parameters."""
    if cfg.TYPE_RETRIEVAL == 'dense':
        # Execute the query
        response_rag = rag_pipeline.run(
            {"text_embedder": {"text": query},
             "prompt_builder": {"question": query},
             "answer_builder": {"query": query}
             }
        )
    elif cfg.TYPE_RETRIEVAL == 'sparse':

        response_rag = rag_pipeline.run(
            {"retriever": {"query": query},
             "prompt_builder": {"question": query},
             "answer_builder": {"query": query}
             }
        )
    elif cfg.TYPE_RETRIEVAL == 'hybrid':
        response_rag = rag_pipeline.run(
            {"text_embedder": {"text": query},
             "bm25_retriever": {"query": query},
             "document_joiner": {"top_k": 5},
             "ranker": {"query": query},
             "prompt_builder": {"question": query},
             "answer_builder": {"query": query}
             }
        )
    elif cfg.TYPE_RETRIEVAL == 'no_rag':
        response_rag = response_rag = rag_pipeline.run(
            {"prompt_builder": {"question": query},
             "answer_builder": {"query": query}
             }
        )
    else:
        response_rag = response_rag = rag_pipeline.run(
            {"prompt_builder": {"question": query},
             "answer_builder": {"query": query}
             }
        )

    return response_rag
