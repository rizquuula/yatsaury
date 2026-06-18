# tests/test_difficulty.py
from tests.helpers import make_chunk
from yatsaury.generators.qa import QaGenerator

BASE_RESPONSE = {"pairs": [{"question": "Q?", "answer": "A.", "supporting_quote": "A."}]}


def test_single_difficulty(mock_llm):
    """difficulty=["easy"] makes 1 call with difficulty hint."""
    mock_llm.complete_json.return_value = BASE_RESPONSE
    gen = QaGenerator()
    gen.generate(make_chunk("A."), n=1, llm=mock_llm, difficulty=["easy"])
    assert mock_llm.complete_json.call_count == 1
    call_args = mock_llm.complete_json.call_args[0][0]  # messages list
    assert any("easy" in str(m) for m in call_args)


def test_multi_difficulty(mock_llm):
    """difficulty=["easy","hard"] makes 2 calls and merges results."""
    mock_llm.complete_json.return_value = BASE_RESPONSE
    gen = QaGenerator()
    samples = gen.generate(make_chunk("A."), n=1, llm=mock_llm, difficulty=["easy", "hard"])
    assert mock_llm.complete_json.call_count == 2
    assert len(samples) == 2


def test_no_difficulty(mock_llm):
    """difficulty=None makes 1 call with no difficulty instruction."""
    mock_llm.complete_json.return_value = BASE_RESPONSE
    gen = QaGenerator()
    gen.generate(make_chunk("A."), n=1, llm=mock_llm, difficulty=None)
    assert mock_llm.complete_json.call_count == 1
