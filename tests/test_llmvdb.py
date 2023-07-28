# /tests/test_simple.py

import pytest
from unittest.mock import patch, Mock, MagicMock
from llmvdb.vdb.doc import ToyDoc
from llmvdb import Llmvdb, APIKeyNotFoundError
import numpy as np

class TestLlmvdb:

    """Unit tests for the Llmvdb class"""
    
    @pytest.fixture
    def llm_instance(self):
        """Fixture to create and return an instance of Llmvdb."""
        return Llmvdb(hugging_face=None, workspace='/path/to/workspace')
    
    @pytest.fixture
    def random_vector(self):
        """Fixture to generate and return a random 4096-dimension vector."""
        return np.random.rand(4096).tolist()
    
    @patch('llmvdb.Llmvdb.create_completion')
    @patch('llmvdb.Model.get_embedding')
    @patch('llmvdb.InMemoryExactNNVectorDB.search')
    def test_generate_prompt(self, mock_db_search, mock_get_embedding, mock_create_completion, llm_instance, random_vector):
        # Arrange
        mock_db_search.return_value = [MagicMock(matches=[MagicMock(text="example text")])]
        mock_get_embedding.return_value = random_vector
        mock_create_completion.return_value = "completed prompt"

        prompt = 'example prompt'

        # Act
        llm_instance.generate_prompt(prompt)

        # Assert
        mock_db_search.assert_called_once()
        mock_create_completion.assert_called_once_with(prompt, "example text")