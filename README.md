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

## 🚀 Getting Started

To get started with LLM-Vector-database, you'll need to clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/juicyjung/LLM-Vector-database.git
```

```bash
pip install -r requirements.txt
```

## 💻 Usage

To use LLM-Vector-database, you'll need to run the construct_db.py script to documents embedding and builing vector database. Here's an example of how to do this:

```bash
python construct_db.py --hugging_face 'juicyjung/easylaw_kr_documents'
```

To use the OpenAI Large Language Model (LLM) with the LLM-Vector-database, you'll need to run the open_ai.py script. Here's how to do this:

```bash
python open_ai.py
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
