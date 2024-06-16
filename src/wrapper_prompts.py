"""Contain wrappers of prompt components of tne rag pipeline."""
from haystack.components.builders import PromptBuilder

from .prompts import PROMPT_TEMPLATE


def setup_prompt():
    """Render a prompt template."""
    return PromptBuilder(template=PROMPT_TEMPLATE)
