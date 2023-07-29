"""Fake embedding"""

from .base import LLM


class FakeLLM(LLM):
    """Fake LLM"""

    def call(self, prompt: str, document: str) -> str:
        return "fake llm output"
