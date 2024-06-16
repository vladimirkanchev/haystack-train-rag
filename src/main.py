"""Main entry point for the rag algorithm."""
import argparse
import timeit

import box
from dotenv import find_dotenv
from dotenv import load_dotenv
import yaml

from wrapper_pipelines import setup_rag_sparse_pipeline
from wrapper_pipelines import setup_rag_dense_pipeline
from wrapper_pipelines import setup_rag_hybrid_pipeline

load_dotenv(find_dotenv())

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='What does Rhodes Statue look like?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()
    QUESTION = args.input

    start = timeit.default_timer()

    if cfg.TYPE_RETRIEVAL == 'dense':
        rag_pipeline = setup_rag_dense_pipeline()
        # Execute the query
        json_response = rag_pipeline.run(({"text_embedder":
                                           {"text": QUESTION},
                                           "prompt_builder":
                                           {"question": QUESTION},
                                           }
                                          )
                                         )
        REPLIES = json_response['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        rag_pipeline = setup_rag_sparse_pipeline()
        json_response = rag_pipeline.run(
            {
                "retriever": {"query": QUESTION},
                "prompt_builder": {"question": QUESTION},
            }
        )
        REPLIES = json_response['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'hybrid':
        rag_pipeline = setup_rag_hybrid_pipeline()
        json_response = rag_pipeline.run(
            {
                "text_embedder": {"text": QUESTION},
                "bm25_retriever": {"query": QUESTION},
                "document_joiner": {"top_k": 5},
                "ranker": {"query": QUESTION},
                "prompt_builder": {"question": QUESTION},
                "llm": {"generation_kwargs":
                        {"max_new_tokens": 500}}
            }
        )
        print(json_response)

    else:
        REPLIES = None
    end = timeit.default_timer()
    REPLIES = json_response['llm']['replies'][0]
    ANSWER = 'No answer found'
    if REPLIES:
        ANSWER = REPLIES[0].strip()

    print(f'\nAnswer:\n {ANSWER}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
