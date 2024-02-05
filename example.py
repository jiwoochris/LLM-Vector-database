from llmvdb import Llmvdb
from llmvdb.embedding.openai import OpenAIEmbedding
from llmvdb.llm.langchain import LangChain

embedding = OpenAIEmbedding()
llm = LangChain(instruction="너는 법률 자문을 위한 챗봇이야. 사용자를 위해 먼저 감정적인 공감을 해줘야해.")

your_llm = Llmvdb(
    embedding,
    llm,
    file_path="jiwoochris/easylaw_kr",
    workspace="workspace_path",
    verbose=False,
)

# your_llm.initialize_db()

while True:
    prompt = input("질문을 입력하세요: ")

    response = your_llm.generate_response(prompt)
    print("답변: ", response)


# "월세방을 얻어 자취를 하고 있는데 군대에 가야합니다. 보증금을 돌려받을 수 있을까요?"