# conftest.py — shared fixtures go here
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_llm():
    return MagicMock()
