"""Main entry point for the rag algorithm."""
import os
from pathlib import Path
import sys
import timeit

import box
from dotenv import find_dotenv, load_dotenv
import yaml

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

with open('rag_system/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

from .evaluate import build_rag_eval_report, evaluate_rag
from .inference import run_pipeline
from .rag_pipelines import select_rag_pipeline
from .utils import create_gt_answer_data, create_question_data
from .utils import load_eval_data, save_eval_data

load_dotenv(find_dotenv())


def main():
    """Apply test questions on Q&A system with ground truth evaluation."""
    query_list = create_question_data()
    start = timeit.default_timer()
    rag_answers, retrieved_docs, rag_questions = [], [], []
    gt_answers = create_gt_answer_data()
    curr_rag_pipeline = select_rag_pipeline()

    for curr_query in query_list:
        rag_answer, retrieved_docs = run_pipeline(curr_query,
                                                  curr_rag_pipeline)
        rag_questions.append(curr_query)
        rag_answers.append(rag_answer)
        retrieved_docs.append(retrieved_docs)

    save_eval_data(rag_answers, retrieved_docs)
    rag_answers, retrieved_docs = load_eval_data()
    inputs, results = evaluate_rag(query_list, rag_answers,
                                   gt_answers, retrieved_docs)
    end = timeit.default_timer()
    build_rag_eval_report(inputs, results)
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")


if __name__ == "__main__":
    main()