"""Contain functions to evaluate rag algorithm."""
from haystack.evaluation.eval_run_result import EvaluationRunResult
import pandas as pd

import box
import yaml

from .eval_pipelines import evaluate_gt_pipeline

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def evaluate_rag(query, rag_answers,
                 gt_answers, retrieved_docs):
    """Evaluate rag algorithm with ground truth data."""
    eval_pipeline = evaluate_gt_pipeline()

    results = eval_pipeline.run({
        "faithfulness": {"questions": query,
                         "contexts":
                         retrieved_docs,
                         "predicted_answers": rag_answers},
        "sas_evaluator": {"predicted_answers": rag_answers,
                          "ground_truth_answers": gt_answers}
    })

    inputs = {
        "question": query,
        "contexts": retrieved_docs,
        "answer": gt_answers,
        "predicted_answer": rag_answers,
    }

    print(results)
    print(inputs)
    return inputs, results


def build_rag_eval_report(inputs, results):
    """Build report of evaluation of rag algorithm."""
    evaluation_result = EvaluationRunResult(run_name="hybrid_rag_pipeline",
                                            inputs=inputs, results=results)
    evaluation_result.score_report()
    results_df = evaluation_result.to_pandas()

    top_3 = results_df.nlargest(3, 'sas_evaluator')
    bottom_3 = results_df.nsmallest(3, 'sas_evaluator')
    results_df.to_excel(cfg.REPORT_PATH)
    pd.concat([top_3, bottom_3])
