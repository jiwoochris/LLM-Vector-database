from llmvdb import Llmvdb
from llmvdb.embedding.openai import OpenAIEmbedding
from llmvdb.llm.langchain import LangChain

embedding = OpenAIEmbedding()
llm = LangChain(
    instruction='너는 챗봇이야. 공감을 잘해주고 친절하게 대해줘.'
)

your_llm = Llmvdb(
    embedding,
    llm,
    file_path="data/generated_data.json",
    workspace="workspace_path",
    verbose=False,
)

# your_llm.initialize_db()


answer = your_llm.generate_response("배고파 힘들어...")
print(answer)