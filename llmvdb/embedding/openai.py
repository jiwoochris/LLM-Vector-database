from .base import Embedding
import os
from openai import OpenAI

from dotenv import load_dotenv
from typing import Optional
from ..exceptions import APIKeyNotFoundError

load_dotenv()


class OpenAIEmbedding(Embedding):
    """Open Ai embedding class to implement a Embedding."""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None

        if self.api_token is None:
            raise APIKeyNotFoundError("OpenAI API key is required")

        self.model = "text-embedding-ada-002"  # 1536 size vector

    def get_embedding(self, prompt):
        """_summary_

        Args:
            prompt (str): text for embedding

        Returns:
            vector: sentence_vector
        """
        client = OpenAI(api_key=self.api_token)
        prompt = prompt.replace("\n", " ")
        sentence_vector = (
            client.embeddings.create(input=[prompt], model=self.model).data[0].embedding
        )

        return sentence_vector
