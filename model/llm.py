from haystack.components.generators import OpenAIGenerator

def setup_llm(model_name):
    return OpenAIGenerator(model=model_name)