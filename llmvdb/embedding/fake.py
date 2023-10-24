"""Fake embedding"""

from .base import Embedding
import numpy as np


class FakeEmbedding(Embedding):
    def get_embedding(self, prompt):
        # return 1536-dimension random vector
        sentence_vector = np.random.rand(1536).tolist()

        return sentence_vector
