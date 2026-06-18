# tests/test_bidirectional.py
from tests.helpers import make_qa_sample
from yatsaury.generators.bidirectional import add_inverses, generate_inverse
from yatsaury.models import DatasetType


def test_generate_inverse_basic(mock_llm):
    mock_llm.complete_json.return_value = {
        "question": "What was Khadijah?",
        "answer": "Wife of the Prophet.",
        "supporting_quote": "Wife of the Prophet",
    }
    sample = make_qa_sample(
        question="Who was Khadijah?", answer="Wife of the Prophet.", quote="Wife of the Prophet"
    )
    inverse = generate_inverse(sample, mock_llm)
    assert inverse is not None
    assert inverse.dataset_type == DatasetType.qa
    assert inverse.fact_id == sample.fact_id


def test_generate_inverse_empty_quote_returns_none(mock_llm):
    mock_llm.complete_json.return_value = {
        "question": "Q", "answer": "A", "supporting_quote": ""
    }
    sample = make_qa_sample(question="Q?", answer="A.", quote="A.")
    assert generate_inverse(sample, mock_llm) is None


def test_add_inverses_returns_original_plus_inverse(mock_llm):
    mock_llm.complete_json.return_value = {
        "question": "IQ?", "answer": "IA.", "supporting_quote": "IA."
    }
    sample = make_qa_sample(question="Q?", answer="A.", quote="A.")
    result = add_inverses([sample], mock_llm)
    assert len(result) == 2
    assert result[0] is sample


def test_add_inverses_none_dropped(mock_llm):
    """If generate_inverse returns None, only original is kept."""
    mock_llm.complete_json.return_value = {
        "question": "Q", "answer": "A", "supporting_quote": ""
    }
    sample = make_qa_sample(question="Q?", answer="A.", quote="A.")
    result = add_inverses([sample], mock_llm)
    assert len(result) == 1
