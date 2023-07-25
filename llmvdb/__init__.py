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


from .module.doc import ToyDoc

from docarray import DocList

from vectordb import InMemoryExactNNVectorDB

from datasets import load_dataset

from .module.model import Model

import openai

import os

from .module.exceptions import APIKeyNotFoundError

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

        # Specify your workspace path
        db = InMemoryExactNNVectorDB[ToyDoc](workspace=self.workspace)

        # Download Data from huggingface
        data = load_dataset(hugging_face)["train"]
        print(data)

        # Define model
        model = Model()

        # Index a list of documents with random embeddings
        doc_list = [
            ToyDoc(text=i["documents"], embedding=model.get_embedding(i["documents"]))
            for i in data
        ]
        db.index(inputs=DocList[ToyDoc](doc_list))

        # Save db
        db.persist()

    def generate(self, prompt):
        api_token = os.getenv("OPENAI_API_KEY") or None
        if api_token is None:
            raise APIKeyNotFoundError("OPEN AI API key is required")

        openai.api_key = api_token

        # Specify your workspace path
        db = InMemoryExactNNVectorDB[ToyDoc](workspace=self.workspace)

        # Define model
        model = Model()

        # Perform a search query
        query = ToyDoc(text=prompt, embedding=model.get_embedding(prompt))
        results = db.search(inputs=DocList[ToyDoc]([query]), limit=5)

        input = results[0].matches[0].text

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f'너는 법률 자문을 위한 챗봇이야. 사용자를 위해 먼저 감정적인 공감을 해주고, 다음 문서를 바탕으로 사용자의 질문에 대해 답변해줘. 문서에서 질문에 대한 답변을 찾을 수 없으면 "없음"이라고 답해줘.\n\n### 문서:\n"\n{input}\n"',
                },
                {"role": "user", "content": f"{prompt}"},
            ],
        )

        print(completion)
        print("->")
        print(completion.choices[0].message.content.strip())

        print("\n\n참고 문서 : \n")
        print(input)
