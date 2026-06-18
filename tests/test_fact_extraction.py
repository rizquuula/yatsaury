# tests/test_fact_extraction.py
from tests.helpers import make_chunk
from yatsaury.generators.fact_extraction import Fact, extract_facts, qa_from_fact
from yatsaury.models import DatasetType


def test_extract_facts_basic(mock_llm):
    mock_llm.complete_json.return_value = {
        "facts": [
            {"text": "Muhammad was born in Mecca.", "source_quote": "born in Mecca"},
            {"text": "He received revelation at 40.", "source_quote": "received revelation"},
        ]
    }
    facts = extract_facts(make_chunk("born in Mecca. received revelation"), mock_llm)
    assert len(facts) == 2
    assert all(f.id for f in facts)
    assert facts[0].text == "Muhammad was born in Mecca."


def test_extract_facts_insufficient(mock_llm):
    mock_llm.complete_json.return_value = {"insufficient": True}
    assert extract_facts(make_chunk("x"), mock_llm) == []


def test_extract_facts_drops_empty_quote(mock_llm):
    mock_llm.complete_json.return_value = {
        "facts": [{"text": "Fact.", "source_quote": ""}]
    }
    assert extract_facts(make_chunk("x"), mock_llm) == []


def test_qa_from_fact(mock_llm):
    mock_llm.complete_json.return_value = {
        "pairs": [{"question": "Q?", "answer": "A.", "supporting_quote": "born in Mecca"}]
    }
    fact = Fact(id="f1", text="Born in Mecca.", source_quote="born in Mecca")
    samples = qa_from_fact(fact, make_chunk("born in Mecca"), mock_llm, n=1)
    assert len(samples) == 1
    assert samples[0].fact_id == "f1"
    assert samples[0].dataset_type == DatasetType.qa


def test_qa_from_fact_empty_quote_dropped(mock_llm):
    mock_llm.complete_json.return_value = {
        "pairs": [{"question": "Q?", "answer": "A.", "supporting_quote": ""}]
    }
    fact = Fact(id="f1", text="F.", source_quote="x")
    samples = qa_from_fact(fact, make_chunk("x"), mock_llm)
    assert samples == []
