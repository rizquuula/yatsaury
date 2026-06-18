"""Tests for SummaryGenerator."""
from __future__ import annotations

from unittest.mock import MagicMock

from yatsaury.generators.base import get_generator
from yatsaury.generators.summary import SummaryGenerator
from yatsaury.models import Chunk, DatasetType


def make_chunk() -> Chunk:
    return Chunk(
        id="chk_abc12345_0000",
        doc_id="abc12345",
        text="The Prophet Muhammad (peace be upon him) was born in Makkah in 570 CE.",
        token_count=20,
        char_span=(0, 70),
        ordinal=0,
    )


class TestSummaryGenerator:
    def test_returns_sample_on_valid_response(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "summary": "S.",
            "key_points": ["p1"],
            "supporting_quote": "S.",
        }
        gen = SummaryGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert len(samples) == 1
        s = samples[0]
        assert s.dataset_type == DatasetType.summary
        assert s.payload["summary"] == "S."
        assert s.payload["key_points"] == ["p1"]
        assert s.payload["passage"] == chunk.text
        assert s.supporting_quote == "S."
        assert s.source_text == chunk.text

    def test_insufficient_returns_empty(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {"insufficient": True}
        gen = SummaryGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert samples == []

    def test_missing_supporting_quote_returns_empty(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "summary": "S.",
            "key_points": ["p1"],
            # no "supporting_quote" key
        }
        gen = SummaryGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert samples == []

    def test_empty_supporting_quote_returns_empty(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "summary": "S.",
            "key_points": ["p1"],
            "supporting_quote": "",
        }
        gen = SummaryGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert samples == []

    def test_sample_chunk_id_matches(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "summary": "S.",
            "key_points": [],
            "supporting_quote": "born in Makkah",
        }
        samples = SummaryGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].chunk_id == chunk.id

    def test_dataset_type_attribute(self):
        assert SummaryGenerator().dataset_type == "summary"


class TestSummaryGeneratorRegistry:
    def test_get_summary_generator(self):
        gen = get_generator("summary")
        assert gen.dataset_type == "summary"
