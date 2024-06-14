"""Main entry point for the rag algorithm."""
import argparse
import timeit

from dotenv import load_dotenv

from wrapper import setup_rag_pipeline

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='What does Rhodes Statue look like?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    rag_pipeline = setup_rag_pipeline()
    QUESTION = args.input

    # Execute the query
    json_response = rag_pipeline.run(({"text_embedder": {"text": QUESTION},
                                       "prompt_builder":
                                       {"question": QUESTION},
                                       }
                                      ))
    end = timeit.default_timer()

    replies = json_response['llm']['replies']
    ANSWER = 'No answer found'
    if replies:
        ANSWER = replies[0].strip()

    print(f'\nAnswer:\n {ANSWER}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
