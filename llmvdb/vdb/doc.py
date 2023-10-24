from docarray import BaseDoc
from docarray.typing import NdArray


class ToyDoc(BaseDoc):
    text: str = ""
    embedding: NdArray[1536]  # NdArray[4096], 1536 for open ai
    related_to: list
