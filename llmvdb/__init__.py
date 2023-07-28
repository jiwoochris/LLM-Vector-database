# -*- coding: utf-8 -*-
"""
This module includes the implementation of basis llm-vector-database class with methods to run
the LLMs models based on vector-database. Following LLMs are implemented so far.

Example:

    This module is the Entry point of the `llm-vector-database` package. Following is an example
    of how to use this Class.

    ```python
    from llmvdb import Llmvdb

    llm_config = "config"

    your_llm = Llmvdb('juicyjung/easylaw_kr_documents')

    answer = your_llm.predict("aaa")

    ```
"""


from .vdb.doc import ToyDoc

from docarray import DocList

from vectordb import InMemoryExactNNVectorDB

from datasets import load_dataset

from .embedding.model import Model

import openai

import os

from .llm.exceptions import APIKeyNotFoundError

from .helpers.ineterface import Interface


class Llmvdb(Interface):
    def __init__(
        self,
        llm=None,
        conversational=False,
        verbose=False,
        enable_cache=True,
        enable_logging=True,
        hugging_face=None,
        workspace=None,
    ):
        self.workspace = workspace
        self.hugging_face = hugging_face
        self.model = Model()
        self.db = self.initialize_db()

    def initialize_db(self):
        # Specify your workspace path
        db = InMemoryExactNNVectorDB[ToyDoc](workspace=self.workspace)

        if self.hugging_face is None:
            return db

        else:
            # Download Data from huggingface
            data = load_dataset(self.hugging_face)["train"]

            # Index a list of documents with random embeddings
            doc_list = [
                ToyDoc(
                    text=i["documents"],
                    embedding=self.model.get_embedding(i["documents"]),
                )
                for i in data
            ]
            db.index(inputs=DocList[ToyDoc](doc_list))

            # Save db
            db.persist()

            return db

    def generate_prompt(self, prompt):
        # Perform a search query
        query = ToyDoc(text=prompt, embedding=self.model.get_embedding(prompt))
        results = self.db.search(inputs=DocList[ToyDoc]([query]), limit=5)

        input = results[0].matches[0].text

        completion = self.create_completion(prompt, input)

        print(completion)
        print("->")
        print(completion.choices[0].message.content.strip())

        print("\n\n참고 문서 : \n")
        print(input)

    def create_completion(self, prompt, input):
        api_token = os.getenv("OPENAI_API_KEY") or None
        if api_token is None:
            raise APIKeyNotFoundError("OPEN AI API key is required")

        openai.api_key = api_token

        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f'너는 법률 자문을 위한 챗봇이야. 사용자를 위해 먼저 감정적인 공감을 해주고, 다음 문서를 바탕으로 사용자의 질문에 대해 답변해줘. 문서에서 질문에 대한 답변을 찾을 수 없으면 "없음"이라고 답해줘.\n\n### 문서:\n"\n{input}\n"',
                },
                {"role": "user", "content": f"{prompt}"},
            ],
        )
