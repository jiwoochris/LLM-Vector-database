from .base import Embedding
import os
import openai
from dotenv import load_dotenv
from typing import Optional
from ..exceptions import APIKeyNotFoundError

load_dotenv()



class OpenAIEmbedding(Embedding):
    def __init__(
        self,
        api_token: Optional[str] = None
    ):
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None

        if self.api_token is None:
            raise APIKeyNotFoundError("OpenAI API key is required")

        openai.api_key = self.api_token
        
        self.model="text-embedding-ada-002"     # 1536 size vector

    
    def get_embedding(self, prompt):
        prompt = prompt.replace("\n", " ")
        sentence_vector = openai.Embedding.create(input = [prompt], model=self.model)['data'][0]['embedding']
        
        return sentence_vector