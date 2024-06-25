"""Main entry point for the rag algorithm."""
import timeit

import box
from dotenv import find_dotenv, load_dotenv
import yaml

from src.evaluate import evaluate_rag, build_rag_eval_report
from src.inference import run_pipeline
from src.rag_pipelines import select_rag_pipeline
from src.utils import create_gt_answer_data, create_question_data
from src.utils import load_eval_data, save_eval_data

load_dotenv(find_dotenv())

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    QUERY_LIST = create_question_data()
    start = timeit.default_timer()
    rag_answers, retrieved_docs, rag_questions = [], [], []
    gt_answers = create_gt_answer_data()
    curr_rag_pipeline = select_rag_pipeline()

    for curr_query in QUERY_LIST:
        rag_answer, retrieved_docs = run_pipeline(curr_query,
                                                  curr_rag_pipeline)
        rag_questions.append(curr_query)
        rag_answers.append(rag_answer)
        retrieved_docs.append(retrieved_docs)

    save_eval_data(rag_answers, retrieved_docs)
    rag_answers, retrieved_docs = load_eval_data()
    inputs, results = evaluate_rag(QUERY_LIST, rag_answers,
                                   gt_answers, retrieved_docs)
    end = timeit.default_timer()
    build_rag_eval_report(inputs, results)
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
