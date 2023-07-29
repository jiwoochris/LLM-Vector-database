from abc import abstractmethod
from ..exceptions import MethodNotImplementedError

class LLM:
    """Base class to implement a new LLM."""

    def is_supported_llm(self) -> bool:
        return True

    @abstractmethod
    def call(self, prompt : str, document : str) -> str:
        """
        Execute the LLM with given prompt.

        Args:
            instruction (Prompt): Prompt

        Raises:
            MethodNotImplementedError: Call method has not been implemented
        """
        raise MethodNotImplementedError("Call method has not been implemented")