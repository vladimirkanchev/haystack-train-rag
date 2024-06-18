"""Contain evaluation haystack pipelines of the rag algorithm."""
from haystack import Pipeline
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator

import box
import yaml

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def evaluate_gt_pipeline():
    """Build basic evaluation haystack pipeline with ground truth data."""
    eval_pipeline = Pipeline()
    #eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas_evaluator",
                                SASEvaluator(model=cfg.EMBEDDINGS))

    return eval_pipeline
