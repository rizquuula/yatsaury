"""Tests for InstructionGenerator."""
from __future__ import annotations

from unittest.mock import MagicMock

from yatsaury.generators.base import get_generator
from yatsaury.generators.instruction import InstructionGenerator
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


class TestInstructionGenerator:
    def test_returns_sample_on_valid_response(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "triples": [
                {
                    "instruction": "Where was the Prophet born?",
                    "input": "",
                    "output": "The Prophet was born in Makkah.",
                    "supporting_quote": "born in Makkah in 570 CE",
                }
            ]
        }
        gen = InstructionGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert len(samples) == 1
        s = samples[0]
        assert s.dataset_type == DatasetType.instruction
        assert s.chunk_id == chunk.id
        assert s.payload["instruction"] == "Where was the Prophet born?"
        assert s.payload["output"] == "The Prophet was born in Makkah."
        assert s.supporting_quote == "born in Makkah in 570 CE"
        assert s.source_text == chunk.text

    def test_insufficient_returns_empty(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {"insufficient": True}
        gen = InstructionGenerator()
        samples = gen.generate(chunk, n=3, llm=mock_llm)
        assert samples == []

    def test_empty_supporting_quote_dropped(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "triples": [
                {
                    "instruction": "I?",
                    "input": "",
                    "output": "O.",
                    "supporting_quote": "",  # empty → drop
                },
                {
                    "instruction": "I2?",
                    "input": "",
                    "output": "O2.",
                    "supporting_quote": "born in Makkah",  # valid
                },
            ]
        }
        gen = InstructionGenerator()
        samples = gen.generate(chunk, n=2, llm=mock_llm)
        assert len(samples) == 1
        assert samples[0].payload["instruction"] == "I2?"

    def test_sample_chunk_id_matches(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "triples": [
                {
                    "instruction": "I?",
                    "input": "",
                    "output": "O.",
                    "supporting_quote": "born in Makkah",
                }
            ]
        }
        samples = InstructionGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].chunk_id == chunk.id

    def test_payload_has_required_keys(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "triples": [
                {
                    "instruction": "I?",
                    "input": "ctx",
                    "output": "O.",
                    "supporting_quote": "born in Makkah",
                }
            ]
        }
        samples = InstructionGenerator().generate(chunk, n=1, llm=mock_llm)
        payload = samples[0].payload
        assert "instruction" in payload
        assert "input" in payload
        assert "output" in payload

    def test_dataset_type_attribute(self):
        assert InstructionGenerator().dataset_type == "instruction"


class TestInstructionGeneratorRegistry:
    def test_get_instruction_generator(self):
        gen = get_generator("instruction")
        assert gen.dataset_type == "instruction"
