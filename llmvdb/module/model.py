# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXModel
import torch


class Model():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
        self.model = InputEmbeddingModel.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")

    def get_embedding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        embedding = self.model(inputs.input_ids)    # ['last_hidden_state']
        
        # Calculate the sentence vector by averaging the hidden states across the sequence length dimension
        sentence_vector = torch.mean(embedding, dim=1)
        
        print(prompt)
        
        return sentence_vector
    
    
    
from transformers import GPTNeoXModel, GPTNeoXConfig

import torch
from typing import List, Optional, Tuple, Union

class InputEmbeddingModel(GPTNeoXModel):

    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) :

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)
            
        return inputs_embeds