"""Build generators for separate llm models."""
from haystack.components.generators import OpenAIGenerator

def setup_llm(model_name):
    """Build generator to call open AI model."""
    return OpenAIGenerator(model=model_name)
