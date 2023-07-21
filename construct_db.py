import argparse

from module.doc import ToyDoc

from docarray import DocList

from vectordb import InMemoryExactNNVectorDB

from datasets import load_dataset

from module.model import Model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hugging_face', default='juicyjung/easylaw_kr_documents', type=str, help="hugging face dataset")  # required=True, 
    parser.add_argument('--workspace', default='workspace_path', type=str, help="workspace path for vector database")
    
    opt = parser.parse_args()

    # Specify your workspace path
    db = InMemoryExactNNVectorDB[ToyDoc](workspace=opt.workspace)

    # Download Data from huggingface
    data = load_dataset(opt.hugging_face)['train']
    print(data)

    # Define model
    model = Model()


    # Index a list of documents with random embeddings
    doc_list = [ToyDoc(text = i['documents'], embedding = model.get_embedding(i['documents'])) for i in data]
    db.index(inputs=DocList[ToyDoc](doc_list))

    # Save db
    db.persist()