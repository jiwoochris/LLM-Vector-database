import argparse

import openai

import os

from module.exceptions import APIKeyNotFoundError

from module.prompter import Prompter

from module.doc import ToyDoc

from module.model import Model

from docarray import DocList

from vectordb import InMemoryExactNNVectorDB



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--template_name', default='law', type=str, help="template_name")  # required=True, 
    parser.add_argument('--workspace', default='workspace_path', type=str, help="workspace path for vector database")
    
    opt = parser.parse_args()
    
    

    api_token = os.getenv("OPENAI_API_KEY") or None
    if api_token is None:
        raise APIKeyNotFoundError("OPEN AI API key is required")

    openai.api_key = api_token


    prompter = Prompter(opt.template_name, verbose=True)
    
    instruction = "질병 때문에 회사에 안나갔는데, 무단결근이라고 해고당했습니다. 이래도 되는거 맞나요?"
    
    
    # Specify your workspace path
    db = InMemoryExactNNVectorDB[ToyDoc](workspace=opt.workspace)
    
    # Define model
    model = Model()

    # Perform a search query
    query = ToyDoc(text = instruction, embedding = model.get_embedding(instruction))
    results = db.search(inputs=DocList[ToyDoc]([query]), limit=5)
    
    input = results[0].matches[0].text
    
    # prompt = prompter.generate_prompt(instruction, input)
    

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 법률 자문을 위한 챗봇이야."},
            {"role": "user", "content": f"{instruction}"}
        ]
    )
    
    print(completion)
    print("->")
    print(completion.choices[0].message.content.strip())
    
    print("참고 문서 : \n")
    print(input)