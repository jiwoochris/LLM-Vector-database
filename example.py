from llmvdb import Llmvdb

llm_config = "config"

your_llm = Llmvdb(
    hugging_face="juicyjung/easylaw_kr_documents", workspace="workspace_path"
)

answer = your_llm.generate_prompt("넌 누가 만들었어?")
