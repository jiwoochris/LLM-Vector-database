from .base import LLM
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, pipeline

# from transformers import AutoTokenizer


class Kullm(LLM):
    """KULLM (구름): Korea University Large Language Model"""

    def __init__(
        self,
        model_name: str = "nlpai-lab/kullm-polyglot-5.8b-v2",
        instruction: Optional[str] = None,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device="cuda", non_blocking=True)
        model.eval()

        self.pipe = pipeline(
            "text-generation", model=model, tokenizer=model_name, device=0
        )

        self.instruction = instruction

    def call(self, prompt: str, document: str) -> str:
        """
        Call the KULLM.
        """

        prompt = f"""아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

### 명령어:
{self.instruction} 다음 문서를 바탕으로 사용자의 질문에 대해 답변해줘. 문서에서 질문에 대한 답변을 찾을 수 없으면 "없음"이라고 답해줘.

### 문서:
{document}

### 입력:
{prompt}

### 응답:
"""

        output = self.pipe(
            prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2
        )

        # print(output)

        s = output[0]["generated_text"]
        result = s.split(self.template["response_split"])[1].strip()

        return result
