# /tests/test_simple.py

import pytest
from unittest.mock import patch, Mock, MagicMock
from llmvdb.vdb.doc import ToyDoc
from llmvdb import Llmvdb, APIKeyNotFoundError
import numpy as np

class TestLlmvdb:

    """Unit tests for the Llmvdb class"""
    def test_add():
        assert 2 == 2