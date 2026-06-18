"""Instruction/input/output triple generator."""
from __future__ import annotations

from uuid import uuid4

from yatsaury.generators.base import register_generator
from yatsaury.llm.client import LLMClient
from yatsaury.llm.prompts import instruction_generation_prompt
from yatsaury.models import Chunk, Citation, DatasetType, Sample


class InstructionGenerator:
    """Generate instruction/input/output triples from a chunk using an LLM."""

    dataset_type = "instruction"

    def generate(self, chunk: Chunk, n: int, llm: LLMClient, **kwargs) -> list[Sample]:
        """Call the LLM and convert response triples into Sample objects.

        - Returns [] if LLM responds with {"insufficient": true}
        - Drops triples with empty supporting_quote
        """
        messages = instruction_generation_prompt(chunk.text, n=n)
        response = llm.complete_json(messages)

        if response.get("insufficient"):
            return []

        triples = response.get("triples", [])
        samples: list[Sample] = []

        for triple in triples:
            instruction = triple.get("instruction", "")
            input_text = triple.get("input", "")
            output = triple.get("output", "")
            supporting_quote = triple.get("supporting_quote", "")

            if not supporting_quote:
                continue

            samples.append(
                Sample(
                    id=uuid4().hex,
                    chunk_id=chunk.id,
                    dataset_type=DatasetType.instruction,
                    payload={"instruction": instruction, "input": input_text, "output": output},
                    source_text=chunk.text,
                    supporting_quote=supporting_quote,
                    source_citation=Citation(title="", source_uri=""),
                )
            )

        return samples


# Register at import time
register_generator(InstructionGenerator())
