"""Build generators for separate llm models."""
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

import box
from dotenv import load_dotenv, find_dotenv
import yaml

load_dotenv(find_dotenv())

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_single_llm(model_name: str) -> None:
    """Build single llm model for RAG algorithm."""
    if cfg.LLM_TYPE == 'openai':
        return OpenAIGenerator(model=model_name)
    if cfg.LLM_TYPE == 'opensource':
        return HuggingFaceTGIGenerator(
            model=model_name,
            token=Secret.from_env_var("HF_API_TOKEN"))
    return None
