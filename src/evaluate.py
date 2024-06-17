"""Contain functions to evaluate rag algorithm."""
from haystack.evaluation.eval_run_result import EvaluationRunResult
import pandas as pd

from .eval_pipelines import evaluate_gt_pipeline


def evaluate_rag(query, rag_answers,
                 gt_answers, retrieved_docs):
    """Evaluate rag algorithm with ground truth data."""
    eval_pipeline = evaluate_gt_pipeline()
    results = eval_pipeline.run({
        "doc_mrr_evaluator": {"ground_truth_documents":
                              list([ans] for ans in gt_answers),
                              "retrieved_documents": retrieved_docs},
        "faithfulness": {"questions": list(query),
                         "contexts": list([d]
                                          for d in retrieved_docs),
                         "predicted_answers": rag_answers},
        "sas_evaluator": {"predicted_answers": rag_answers,
                          "ground_truth_answers": list(gt_answers)}
    })

    inputs = {
        "question": list(query),
        "contexts": list([d] for d in retrieved_docs),
        "answer": list(gt_answers),
        "predicted_answer": rag_answers,
    }

    return inputs, results


def build_rag_eval_report(inputs, results):
    """Build report of evaluation of rag algorithm."""
    evaluation_result = EvaluationRunResult(run_name="pubmed_rag_pipeline",
                                            inputs=inputs, results=results)
    evaluation_result.score_report()
    results_df = evaluation_result.to_pandas()
    print(results_df)

    top_3 = results_df.nlargest(3, 'sas_evaluator')
    bottom_3 = results_df.nsmallest(3, 'sas_evaluator')
    pd.concat([top_3, bottom_3])
