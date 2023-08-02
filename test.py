from llmvdb import Llmvdb
from llmvdb.embedding.model import HuggingFaceEmbedding
from llmvdb.embedding.fake import FakeEmbedding

from llmvdb.llm.openai import OpenAI
from llmvdb.llm.fake import FakeLLM

embedding = FakeEmbedding()
llm = FakeLLM()

your_llm = Llmvdb(
    embedding,
    llm,
    hugging_face="juicyjung/easylaw_kr",
    workspace="workspace_path",  # "juicyjung/easylaw_kr_documents"
)

answer = your_llm.generate_prompt("월세방을 얻어 자취를 하고 있는데 군대에 가야합니다. 보증금을 돌려받을 수 있을까요?")
print(answer)
