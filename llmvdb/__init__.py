from .vdb.doc import ToyDoc
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB
from .helpers.ineterface import Interface
from typing import Optional


from .vdb.customdataset import CustomDataset


class Llmvdb(Interface):
    def __init__(
        self,
        embedding=None,
        llm=None,
        verbose: bool = False,
        file_path=None,
        workspace: Optional[str] = None,
        threshold: float = 0.8,
        top_k: int = 3,
    ):
        self.embedding = embedding
        self.llm = llm

        self.verbose = verbose
        self.llm.verbose = self.verbose

        self.workspace = workspace
        self.file_path = file_path

        self.threshold = threshold
        self.top_k = top_k

        # Specify your workspace path
        self.db = InMemoryExactNNVectorDB[ToyDoc](workspace=self.workspace)

    def initialize_db(self):
        dataset = CustomDataset(self.file_path).documents_data

        # Index a list of documents with random embeddings
        doc_list = [
            ToyDoc(text=t['text'], embedding=self.embedding.get_embedding(t['text']))
            for t in dataset
        ]

        self.db.index(inputs=DocList[ToyDoc](doc_list))

        # Save db
        self.db.persist()

    def control_threshold(self, value):
        self.threshold = value

    def control_top_k(self, value):
        self.top_k = value

    def retrieve_document(self, prompt):
        # Perform a search query
        query = ToyDoc(
            text=prompt, embedding=self.embedding.get_embedding(prompt), related_to=[]
        )
        results = self.db.search(inputs=DocList[ToyDoc]([query]), limit=5)

        if self.verbose:
            print(results[0])

        input_document = ""
        over_threshold_indices = [
            i for i, value in enumerate(results[0].scores) if value > self.threshold
        ]

        # 만약 threshold 0.8을 넘는게 있고 그 개수가 k개보다 적다면 전부 retrieve
        if 1 <= len(over_threshold_indices) < self.top_k:
            for index in over_threshold_indices:  # top-k (k=3)
                input_document += (
                    "#문서" + str(index) + "\n" + results[0].matches[index].text + "\n"
                )

        # 만약 threshold 0.8을 넘는게 있고 그 개수가 k개보다 많다면 top-k만 retrieve
        elif len(over_threshold_indices) >= self.top_k:
            for index in range(self.top_k):  # top-k (k=3)
                input_document += (
                    "#문서" + str(index) + "\n" + results[0].matches[index].text + "\n"
                )

        # 만약 threshold 0.8을 넘는게 없다면 top-1만
        elif len(over_threshold_indices) == 0:
            input_document += "#문서\n" + results[0].matches[0].text + "\n"

        if self.verbose:
            print("아래 문서를 참고합니다: \n")
            print(input_document)

        return input_document

    def generate_response(self, prompt):
        input_document = self.retrieve_document(prompt)

        completion = self.llm.call(prompt, input_document)

        return completion

    def flush_history(self):
        """Flush chat history memory and re-initialize the conversation"""
        self.llm.history_memory = self.llm.initial_history_memory
