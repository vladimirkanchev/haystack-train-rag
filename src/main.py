"""Main entry point for the rag algorithm."""
import argparse
import timeit
from haystack.components.builders.answer_builder import AnswerBuilder

from dotenv import load_dotenv
import box, yaml

from wrapper import setup_rag_sparse_pipeline, setup_rag_dense_pipeline

load_dotenv()

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
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
                                        ))
        replies = json_response['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        rag_pipeline = setup_rag_sparse_pipeline()
        json_response = rag_pipeline.run(
            {
                "retriever": {"query": QUESTION},
                "prompt_builder": {"question": QUESTION},
            }
        )
        replies = json_response['llm']['replies']

    end = timeit.default_timer()

    
    ANSWER = 'No answer found'
    if replies:
        ANSWER = replies[0].strip()

    print(f'\nAnswer:\n {ANSWER}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
