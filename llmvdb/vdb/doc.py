from docarray import BaseDoc
from docarray.typing import NdArray


class ToyDoc(BaseDoc):
    text: str = ""
    embedding: NdArray[4096]  # NdArray[4096]


# class ToyDoc(BaseDoc):
#     text: str = ""
#     embedding_size: int = 4096  # This sets the size for all instances

#     def __init__(self, text=""):
#         self.text = text
#         self.embedding = NdArray(self.embedding_size)
