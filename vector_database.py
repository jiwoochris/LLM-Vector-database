from docarray import BaseDoc
from docarray.typing import NdArray

class ToyDoc(BaseDoc):
  text: str = ''
  embedding: NdArray[128]


from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

# Specify your workspace path
db = InMemoryExactNNVectorDB[ToyDoc](workspace='./workspace_path')




# Index a list of documents with random embeddings
doc_list = [ToyDoc(text=f'toy doc {i}', embedding=np.random.rand(128)) for i in range(1000)]
db.index(inputs=DocList[ToyDoc](doc_list))





# Perform a search query
query = ToyDoc(text='query', embedding=np.random.rand(128))
results = db.search(inputs=DocList[ToyDoc]([query]), limit=10)


db.persist()

# Print out the matches
for m in results[0].matches:
  print(m)