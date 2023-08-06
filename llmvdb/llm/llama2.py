from .base import LLM
from typing import Optional, List
from transformers import AutoTokenizer, pipeline
import torch
from ..exceptions import MethodNotImplementedError


class Llama2(LLM):
    """Meta Llama2 LLM in Korean"""

    _supported_chat_models: List[str] = [
        "beomi/kollama-7b",
        "beomi/kollama-13b",
        "beomi/kollama-33b",
        "beomi/llama-2-ko-7b",
    ]

    model: str = "beomi/llama-2-ko-7b"

    def __init__(self, instruction: Optional[str] = None):
        self.instruction = instruction

        _tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.eos_token_id = _tokenizer.eos_token_id
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def call(self, prompt: str = None, document: str = None) -> str:
        """
        Call the Llama2 LLM.
        Args:
            instruction (prompt, document): Instruction to pass
        Raises:
            MethodNotImplementedError: Not Implemented Error
        Returns:
            str: Response
        """

        if self.model in self._supported_chat_models:
            response = self.pipeline(
                f"{self.instruction}\n\n{prompt}\n",
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=self.eos_token_id,
                max_length=200,
            )

        else:
            raise MethodNotImplementedError("Not implemented yet")

        return response[0]["generated_text"]
