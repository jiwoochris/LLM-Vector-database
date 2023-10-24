from abc import abstractmethod
from ..exceptions import MethodNotImplementedError


class Embedding:
    """Base class to implement a new Embedding."""

    def is_supported_embedding(self) -> bool:
        return True

    @abstractmethod
    def get_embedding(self, prompt: str):
        """
        Get embedding with given prompt.

        Args:
            instruction (Prompt): Prompt

        Raises:
            MethodNotImplementedError: Call method has not been implemented
        """
        raise MethodNotImplementedError("Call method has not been implemented")
