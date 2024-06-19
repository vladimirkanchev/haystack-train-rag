"""Main entry point for the rag algorithm."""
import timeit

import box
from dotenv import find_dotenv, load_dotenv
from haystack import Pipeline
import yaml

from src.rag_pipelines import select_rag_pipeline


from src.evaluate import evaluate_rag, build_rag_eval_report
from src.utils import create_gt_answer_data, create_question_data
from src.utils import load_eval_data, save_eval_data

load_dotenv(find_dotenv())

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


if __name__ == "__main__":
    QUERY_LIST = create_question_data()
    start = timeit.default_timer()
    rag_answers, retrieved_docs, rag_questions = [], [], []
    gt_answers = create_gt_answer_data()
    curr_rag_pipeline = select_rag_pipeline()

    for curr_query in QUERY_LIST:
        response = run_pipeline(curr_query, curr_rag_pipeline)
        rag_questions.append(curr_query)
        rag_answers.append(response["answer_builder"]["answers"][0].data)
        retrieved_docs.append(response["answer_builder"][
            "answers"][0].documents)
    save_eval_data(rag_answers, retrieved_docs)
    rag_answers, retrieved_docs = load_eval_data()
    inputs, results = evaluate_rag(QUERY_LIST, rag_answers,
                                   gt_answers, retrieved_docs)
    end = timeit.default_timer()
    build_rag_eval_report(inputs, results)
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
