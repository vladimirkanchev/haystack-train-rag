"""Main entry point for the rag algorithm."""
import argparse
import timeit

import box
from dotenv import find_dotenv
from dotenv import load_dotenv
import yaml

from .src.rag_pipelines import setup_rag_sparse_pipeline
from .src.rag_pipelines import setup_rag_dense_pipeline
from .src.rag_pipelines import setup_rag_hybrid_pipeline

from .src.utils import create_gt_answer_data, create_question_data
from .src.evaluate import evaluate_rag, build_rag_eval_report

load_dotenv(find_dotenv())

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='What does Rhodes Statue look like?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()
    QUESTION = create_question_data()
    start = timeit.default_timer()
    rag_answers = []
    retrieved_docs = []
    rag_questions = []
    rag_questions.append(QUESTION)
    gt_answers = create_gt_answer_data()
    if cfg.TYPE_RETRIEVAL == 'dense':
        rag_pipeline = setup_rag_dense_pipeline()
        # Execute the query
        response_rag = rag_pipeline.run(({"text_embedder":
                                          {"text": QUESTION},
                                          "prompt_builder":
                                          {"question": QUESTION},
                                          }
                                         )
                                        )
        REPLIES = response_rag['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        rag_pipeline = setup_rag_sparse_pipeline()
        response_rag = rag_pipeline.run(
            {
                "retriever": {"query": QUESTION},
                "prompt_builder": {"question": QUESTION},
            }
        )
        REPLIES = response_rag['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'hybrid':
        rag_pipeline = setup_rag_hybrid_pipeline()
        response_rag = rag_pipeline.run(
            {
                "text_embedder": {"text": QUESTION},
                "bm25_retriever": {"query": QUESTION},
                "document_joiner": {"top_k": 5},
                "ranker": {"query": QUESTION},
                "prompt_builder": {"question": QUESTION},
                "answer_builder": {"query": QUESTION}
            }
        )
    else:
        response_rag = None
    end = timeit.default_timer()
    # REPLIES = response_rag['llm']['replies'][0]
    rag_answers.append(response_rag["answer_builder"]["answers"][0].data)
    retrieved_docs.append(response_rag["answer_builder"][
        "answers"][0].documents)
    rag_questions.append(QUESTION)
    print(type(gt_answers))
    print(type(gt_answers[0]))
    print(gt_answers[0])
    inputs, results = evaluate_rag(QUESTION, rag_answers,
                                   gt_answers, retrieved_docs)
    build_rag_eval_report(inputs, results)
    ANSWER = 'No answer found'
    # if REPLIES:
    #    ANSWER = REPLIES[0].strip()
    print(f'\nAnswer:\n {ANSWER}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
