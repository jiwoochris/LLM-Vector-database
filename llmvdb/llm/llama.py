from .base import LLM
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers import AutoTokenizer


class Llama(LLM):
    """Llama2 fine-tuned version"""

    def __init__(
        self,
        model_name: str = "maywell/Synatra-10.7B-v0.4",
        instruction: Optional[str] = None,
    ):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.instruction = instruction

        self.system_message = f"{self.instruction} 다음 문서를 바탕으로 사용자의 질문에 대해 답변해줘. 사용자의 질문을 자세히 분석하고 문서에서 질문에 대한 답변을 찾을 수 없으면 문서를 절대 참고하지 마."


    def call(self, prompt: str, document: str) -> str:
        """
        Call the KULLM.
        """

        content = f"{self.system_message}\n\n{prompt}\n\n### 그냥 참고용(질문과 관련 없으면 절대 참고하지 않기):\n{document}\n"

        messages = [
            {"role": "user", "content": content}
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        model.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)

        return decoded[0]
