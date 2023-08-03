from llmvdb import Llmvdb
from llmvdb.embedding.model import HuggingFaceEmbedding
from llmvdb.llm.openai import OpenAI

embedding = HuggingFaceEmbedding()
llm = OpenAI(instruction="너는 법률 자문을 위한 챗봇이야. 사용자를 위해 감정적인 공감을 해준 이후 답변을 해줘.")

your_llm = Llmvdb(
    embedding,
    llm,
    hugging_face="juicyjung/easylaw_kr",
    workspace="workspace_path",  # "juicyjung/easylaw_kr"
)

answer = your_llm.generate_prompt("월세방을 얻어 자취를 하고 있는데 군대에 가야합니다. 보증금을 돌려받을 수 있을까요?")
print(answer)
