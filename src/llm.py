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


def setup_openai_llm(model_name):
    """Build generator to call open AI model."""
    return OpenAIGenerator(model=model_name)


def setup_huggface_llm(model_name):
    """Build generator to call hug"""
    return HuggingFaceTGIGenerator(model=cfg.LLM_MODEL,
                                   token=Secret.from_env_var("HF_API_TOKEN"))


def setup_single_llm(model_name):
    """Build single llm model for RAG algorithm"""
    if cfg.LLM_TYPE == 'openai':
        return setup_openai_llm(model_name)
    elif cfg.LLM_TYPE == 'opensource':
        return setup_huggface_llm(model_name)