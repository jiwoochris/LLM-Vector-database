# 🗄️ LLM-Vector-database

"Don't fine-tune your LLM, **Construct** a vector database."

"There is **No Hallucination** here."

## 🔍 Overview

LLM-Vector-database is a powerful tool that allows you to construct a vector database using sentence embeddings. Instead of fine-tuning your Large Language Model (LLM), this project provides a unique approach to natural language processing and understanding. By embedding sentences into a vector space and constructing a database from these vectors, you can generate text responses based on this database. This makes it an ideal resource for chatbot development and other natural language processing applications.

<img src="https://github.com/juicyjung/LLM-Vector-database/assets/83687471/384a9fe0-00dc-454a-a625-9fa5c22bad11">
<br>
<p align="center">
  <img src="https://github.com/juicyjung/LLM-Vector-database/assets/83687471/66aa6397-38c4-4d49-a298-4a736e102111" width="500">
</p>


## 🌟 Features

- Sentence embedding: Convert your sentences into vector representations.
- Vector database construction: Build a database from your sentence vectors.
- Text generation: Generate text responses based on the vector database.

## 🚀 Quickstart Guide

Follow these steps to get started with the LLM-Vector-database:

**1. Clone the repository** 

Use the following command to clone the repository:

```bash
git clone https://github.com/juicyjung/LLM-Vector-database.git
```

**2. Install the necessary dependencies**

After cloning the repository, navigate into the directory and install the necessary dependencies, vectordb and torch(appropriate version for your environment) by executing:

```bash
pip install poetry
poetry install
```

That's it! You've successfully set up LLM-Vector-database on your machine.

## 💻 Usage

Follow these steps to utilize LLM-Vector-database in your project:

```python
from llmvdb import Llmvdb
from llmvdb.embedding.model import HuggingFaceEmbedding
from llmvdb.llm.openai import OpenAI

embedding = HuggingFaceEmbedding()
llm = OpenAI(instruction="너는 법률 자문을 위한 챗봇이야. 사용자를 위해 먼저 감정적인 공감을 해줘야해.")

your_llm = Llmvdb(
    embedding,
    llm,
    hugging_face="juicyjung/easylaw_kr_documents",
    workspace="workspace_path",
)

answer = your_llm.generate_prompt("월세방을 얻어 자취를 하고 있는데 군대에 가야합니다. 보증금을 돌려받을 수 있을까요?")
print(answer)
```

The above code will return the following:
```
군대에 입대해야 하는 경우에는 임차인이 임대차 계약을 중도해지할 수 있는 사유에 해당하지 않습니다. 따라서, 약정한 기간이 남은 임대차의 경우에는 보증금을 돌려받을 수 없으며, 약정한 기간 동안 월세를 지
급해야 합니다.
```

## 🏆 Advantages of Using LLM-Vector-database

Using the method described above, we were able to significantly address two major issues that arose when fine-tuning LAW-Alpaca.

### 1. Reduced Training Burden

Fine-tuning requires high-performance GPU resources and takes about 5 hours each time based on approximately 2000 data. However, by using a Vector Database, we were able to use the LLM off-the-shelf, which saved costs during training. The process of embedding and constructing the vector database took less than 1 minute, significantly reducing the time and resources required compared to traditional fine-tuning methods.

   
### 2. Solved Hallucination Problem

The advantage of a language model is inference and generation from given language data, not fact searching. Therefore, if you simply ask the LLM a fact-based question, it can produce plausible but false information, regardless of how much fine-tuning has been done. However, when we applied this architecture, we changed the role of the LLM from fact-based questioning to a QA task, preserving the LLM's strength in inference while solving the problem of hallucination.


## 🤝 Contributing

Contributions are welcome! Please check out the todos below, and feel free to open a pull request.

## 📞 Contact

If you have any questions, feel free to reach out to us. We'd be more than happy to assist you!
