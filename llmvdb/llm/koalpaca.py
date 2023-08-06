from .base import LLM
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class KoAlpaca(LLM):
    """Meta Llama2 LLM in Korean"""

    # _supported_chat_models: List[str] = []

    model_name: str = "beomi/KoAlpaca-Polyglot-5.8B"

    def __init__(self, instruction: Optional[str] = None):
        self.instruction = instruction
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device=f"cuda", non_blocking=True)
        model.eval()
        self.pipe = pipeline(
            "text-generation", model=model, tokenizer=self.model_name, device=0
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
        prompt = f"""
        ### 질문: {self.instruction} 다음 문서를 바탕으로 사용자의 질문에 대해 답변해줘. 문서에서 질문에 대한 답변을 찾을 수 없으면 "없음"이라고 답해줘.{prompt}\n\n
        ### 맥락: {document}\n\n
        ### 답변:
        """

        response = self.pipeline(
            f"{self.instruction}\n\n{prompt}\n",
            do_sample=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            eos_token_id=2,
        )

        return response[0]["generated_text"]
