"""Main entry point for the rag algorithm."""
import argparse
import timeit
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]))

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
            {"text_embedder":{"text": query},
                "prompt_builder":{"question": query},
            }
        )
        REPLIES = response_rag['llm']['replies']
    elif cfg.TYPE_RETRIEVAL == 'sparse':
        rag_pipeline = setup_rag_sparse_pipeline()
        response_rag = rag_pipeline.run(
            {"retriever": {"query": query},
             "prompt_builder": {"question": query},
            }
        )
        REPLIES = response_rag['llm']['replies']
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
    query = create_question_data()
    start = timeit.default_timer()
    rag_answers, retrieved_docs, rag_questions = [], [], []
    gt_answers = create_gt_answer_data()
 
    response_rag = run_pipeline(query)

    end = timeit.default_timer()
    
    rag_questions.append(query)
    rag_answers.append(response_rag["answer_builder"]["answers"][0].data)
    retrieved_docs.append(response_rag["answer_builder"][
        "answers"][0].documents)

    inputs, results = evaluate_rag(query, rag_answers,
                                   gt_answers, retrieved_docs)
    build_rag_eval_report(inputs, results)
    ANSWER = 'No answer found'
    # if REPLIES:
    #    ANSWER = REPLIES[0].strip()
    print(f'\nAnswer:\n {ANSWER}')
    print('=' * 50)
    print(f"Time to retrieve answer: {end - start}")
