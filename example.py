from llmvdb import Llmvdb

# Instantiate a embedding
from llmvdb.embedding.model import HuggingFaceEmbedding

# Instantiate a LLM
from llmvdb.llm.openai import OpenAI

embedding = HuggingFaceEmbedding()


llm = OpenAI(instruction="너는 법률 자문을 위한 챗봇이야. 사용자를 위해 먼저 감정적인 공감을 해줘야해.")


your_llm = Llmvdb(
    embedding,
    llm,
    hugging_face=None,
    workspace="workspace_path",  # "juicyjung/easylaw_kr_documents"
)

answer = your_llm.generate_prompt("넌 누가 개발했어?")
