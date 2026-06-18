# tests/test_paraphrases.py
from tests.helpers import make_chunk
from yatsaury.generators.qa import QaGenerator

BASE_RESPONSE = {"pairs": [{"question": "Q?", "answer": "A.", "supporting_quote": "A."}]}
PARAPHRASE_RESPONSE = {"variants": [{"question": "Q2?", "answer": "A2.", "supporting_quote": "A."}]}


def test_paraphrases_same_fact_id(mock_llm):
    """N paraphrases of one Q&A share a fact_id."""
    mock_llm.complete_json.side_effect = [BASE_RESPONSE, PARAPHRASE_RESPONSE]
    chunk = make_chunk("source text A.")
    gen = QaGenerator()
    samples = gen.generate(chunk, n=1, llm=mock_llm, paraphrases=2)
    assert len(samples) == 2
    fact_ids = {s.fact_id for s in samples}
    assert len(fact_ids) == 1
    assert fact_ids != {None}


def test_paraphrases_one_means_no_extra_call(mock_llm):
    """paraphrases=1 makes only 1 LLM call (no paraphrase step)."""
    mock_llm.complete_json.return_value = BASE_RESPONSE
    gen = QaGenerator()
    gen.generate(make_chunk("A."), n=1, llm=mock_llm, paraphrases=1)
    assert mock_llm.complete_json.call_count == 1


def test_paraphrases_base_gets_fact_id(mock_llm):
    """Even with paraphrases=1, base sample has fact_id set."""
    mock_llm.complete_json.return_value = BASE_RESPONSE
    gen = QaGenerator()
    samples = gen.generate(make_chunk("A."), n=1, llm=mock_llm, paraphrases=1)
    assert samples[0].fact_id is not None


def test_different_qa_pairs_different_fact_ids(mock_llm):
    """Two Q&A pairs from the same chunk get different fact_ids."""
    mock_llm.complete_json.return_value = {
        "pairs": [
            {"question": "Q1?", "answer": "A1.", "supporting_quote": "A1."},
            {"question": "Q2?", "answer": "A2.", "supporting_quote": "A2."},
        ]
    }
    gen = QaGenerator()
    samples = gen.generate(make_chunk("A1. A2."), n=2, llm=mock_llm, paraphrases=1)
    assert samples[0].fact_id != samples[1].fact_id
