from docarray import BaseDoc
from docarray.typing import NdArray


class ToyDoc(BaseDoc):
    text: str = ""
    embedding: NdArray[4096]
