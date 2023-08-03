# -*- coding: utf-8 -*-
"""
This module includes the implementation of basis llm-vector-database class with methods to run
the LLMs models based on vector-database. Following LLMs are implemented so far.

Example:

    This module is the Entry point of the `llm-vector-database` package. Following is an example
    of how to use this Class.

    ```from llmvdb import Llmvdb
    from llmvdb.embedding.model import HuggingFaceEmbedding
    from llmvdb.llm.openai import OpenAI

    embedding = HuggingFaceEmbedding()
    llm = OpenAI(instruction="너는 법률 자문을 위한 챗봇이야. 사용자를 위해 감정적인 공감을 해준 이후 답변을 해줘.")

    your_llm = Llmvdb(
        embedding,
        llm,
        hugging_face="juicyjung/easylaw_kr_documents",
        workspace="workspace_path",
    )

    answer = your_llm.generate_prompt("월세방을 얻어 자취를 하고 있는데 군대에 가야합니다. 보증금을 돌려받을 수 있을까요?")
    print(answer)
    ```
"""

from .vdb.doc import ToyDoc
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB
from .vdb.huggingface import HuggingFaceDataset
from .helpers.ineterface import Interface
from typing import Optional


class Llmvdb(Interface):
    db: InMemoryExactNNVectorDB

    def __init__(
        self,
        embedding=None,
        llm=None,
        verbose: bool = False,
        hugging_face=None,
        workspace: Optional[str] = None,
    ):
        self.embedding = embedding
        self.llm = llm
        self.verbose = verbose

        self.workspace = workspace
        self.hugging_face = hugging_face
        self.db = self.initialize_db()

    def initialize_db(self):
        # Specify your workspace path
        db = InMemoryExactNNVectorDB[ToyDoc](workspace=self.workspace)

        if self.hugging_face is None:
            return db

        else:
            dataset = HuggingFaceDataset(self.hugging_face).documents_data

            # Index a list of documents with random embeddings
            doc_list = [
                ToyDoc(
                    text=i["documents"],
                    embedding=self.embedding.get_embedding(i["documents"]),
                )
                for i in dataset
            ]
            db.index(inputs=DocList[ToyDoc](doc_list))

            # Save db
            db.persist()

            return db

    def generate_prompt(self, prompt):
        # Perform a search query
        query = ToyDoc(text=prompt, embedding=self.embedding.get_embedding(prompt))
        results = self.db.search(inputs=DocList[ToyDoc]([query]), limit=5)

        input = results[0].matches[0].text

        completion = self.llm.call(prompt, input)
        respond = completion.choices[0].message.content.strip()

        if self.verbose:
            print("아래 문서를 참고합니다: \n")
            print(input)

        return respond
