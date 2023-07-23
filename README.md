# LLM-Vector-database

"Don't fine-tune your LLM, **Construct** a vector database."

"There is **No Hallucination** here."

## Overview

LLM-Vector-database is a powerful tool that allows you to construct a vector database using sentence embeddings. Instead of fine-tuning your Large Language Model (LLM), this project provides a unique approach to natural language processing and understanding. By embedding sentences into a vector space and constructing a database from these vectors, you can generate text responses based on this database. This makes it an ideal resource for chatbot development and other natural language processing applications.

<img src="https://github.com/juicyjung/LLM-Vector-database/assets/83687471/384a9fe0-00dc-454a-a625-9fa5c22bad11">
<br>
<p align="center">
  <img src="https://github.com/juicyjung/LLM-Vector-database/assets/83687471/66aa6397-38c4-4d49-a298-4a736e102111" width="500">
</p>


## Features

- Sentence embedding: Convert your sentences into vector representations.
- Vector database construction: Build a database from your sentence vectors.
- Text generation: Generate text responses based on the vector database.

## Getting Started

To get started with LLM-Vector-database, you'll need to clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/juicyjung/LLM-Vector-database.git
```

```bash
pip install -r requirements.txt
```

## Usage

To use LLM-Vector-database, you'll need to run the construct_db.py script to documents embedding and builing vector database. Here's an example of how to do this:

```bash
python construct_db.py --hugging_face 'juicyjung/easylaw_kr_documents'
```

To use the OpenAI Large Language Model (LLM) with the LLM-Vector-database, you'll need to run the open_ai.py script. Here's how to do this:

```bash
python open_ai.py
```

## Contact

If you have any questions, feel free to reach out to us. We'd be more than happy to assist you!
