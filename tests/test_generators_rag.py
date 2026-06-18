"""Tests for RagGenerator."""
from __future__ import annotations

from unittest.mock import MagicMock

from yatsaury.generators.base import get_generator
from yatsaury.generators.rag import RagGenerator
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


class TestRagGenerator:
    def test_returns_exactly_one_sample(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        gen = RagGenerator()
        samples = gen.generate(chunk, n=3, llm=mock_llm)
        assert len(samples) == 1

    def test_dataset_type_is_rag(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        gen = RagGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert samples[0].dataset_type == DatasetType.rag

    def test_grounding_score_is_1(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        samples = RagGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].grounding_score == 1.0

    def test_verified_is_true(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        samples = RagGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].verified is True

    def test_no_llm_call(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        RagGenerator().generate(chunk, n=3, llm=mock_llm)
        mock_llm.complete_json.assert_not_called()

    def test_payload_text_equals_chunk_text(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        samples = RagGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].payload["text"] == chunk.text

    def test_chunk_id_matches(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        samples = RagGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].chunk_id == chunk.id

    def test_payload_has_page_and_char_span(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        samples = RagGenerator().generate(chunk, n=1, llm=mock_llm)
        payload = samples[0].payload
        assert "page" in payload
        assert "char_span" in payload
        assert payload["char_span"] == list(chunk.char_span)

    def test_supporting_quote_is_first_200_chars(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        samples = RagGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].supporting_quote == chunk.text[:200]

    def test_dataset_type_attribute(self):
        assert RagGenerator().dataset_type == "rag"


class TestRagGeneratorRegistry:
    def test_get_rag_generator(self):
        gen = get_generator("rag")
        assert gen.dataset_type == "rag"
