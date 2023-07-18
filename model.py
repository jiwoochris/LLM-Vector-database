# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXModel
from llama import Custom_LlamaModel

import torch


class Model():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
        self.model = Custom_LlamaModel.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")

    def get_embedding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        embedding = self.model(inputs.input_ids)    # ['last_hidden_state']
        
        # Calculate the sentence vector by averaging the hidden states across the sequence length dimension
        sentence_vector = torch.mean(embedding, dim=1)
        
        print(prompt)
        
        return sentence_vector


# model = Model()

# prompt = "### 질문: 안녕, 너는 누구니? 나이랑 이름을 알려줘 \n\n### 답변: "

# embdding = model.get_embedding(prompt)

# print(embdding)
# print(embdding.shape)


# exit()



# from datasets import load_dataset

# data = load_dataset("juicyjung/easylaw_kr")

# print(data)

# print(data['train'][0]['instruction'])





# for i in data['train']:
#     embdding = model.get_embedding(i['instruction'])



# answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(answer)
# # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."


# # get_input_embeddings