"""Main entry point for the rag algorithm."""
import timeit

import box
from dotenv import find_dotenv, load_dotenv
import yaml

from src.rag_pipelines import setup_rag_sparse_pipeline
from src.rag_pipelines import setup_rag_dense_pipeline
from src.rag_pipelines import setup_rag_hybrid_pipeline

from src.evaluate import evaluate_rag, build_rag_eval_report
from src.utils import create_gt_answer_data, create_question_data

load_dotenv(find_dotenv())

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def run_pipeline(query):
    """Run rag/no rag pipeline with predifined parameters."""
    if cfg.TYPE_RETRIEVAL == 'dense':
        rag_pipeline = setup_rag_dense_pipeline()
        # Execute the query
        response_rag = rag_pipeline.run(
            {"text_embedder": {"text": query},
                "prompt_builder": {"question": query},
             }
        )
        # curr_reply = response_rag['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        rag_pipeline = setup_rag_sparse_pipeline()
        response_rag = rag_pipeline.run(
            {"retriever": {"query": query},
             "prompt_builder": {"question": query},
             }
        )
        # curr_reply = response_rag['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'hybrid':
        rag_pipeline = setup_rag_hybrid_pipeline()
        response_rag = rag_pipeline.run(
            {"text_embedder": {"text": query},
             "bm25_retriever": {"query": query},
             "document_joiner": {"top_k": 5},
             "ranker": {"query": query},
             "prompt_builder": {"question": query},
             "answer_builder": {"query": query}
             }
        )
    else:
        response_rag = None
    return response_rag


if __name__ == "__main__":
    QUERY = create_question_data()
    start = timeit.default_timer()
    rag_answers, retrieved_docs, rag_questions = [], [], []
    gt_answers = create_gt_answer_data()

    response = run_pipeline(QUERY)

    end = timeit.default_timer()

    rag_questions.append(QUERY)
    rag_answers.append(response["answer_builder"]["answers"][0].data)

    retrieved_docs.append(response["answer_builder"][
        "answers"][0].documents)
    reply = response["answer_builder"]["answers"]
    inputs, results = evaluate_rag(QUERY, rag_answers,
                                   gt_answers, retrieved_docs)
    build_rag_eval_report(inputs, results)
    # if reply:
    #    final_answer = reply[0].strip()
        # print(f'\nAnswer:\n {final_answer}')
    # else:
        # print('\nAnswer:\n No final answer')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
