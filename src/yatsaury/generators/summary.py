"""Summary generator."""
from __future__ import annotations

from uuid import uuid4

from yatsaury.generators.base import register_generator
from yatsaury.llm.client import LLMClient
from yatsaury.llm.prompts import summary_generation_prompt
from yatsaury.models import Chunk, Citation, DatasetType, Sample


class SummaryGenerator:
    """Generate a summary sample from a chunk using an LLM."""

    dataset_type = "summary"

    def generate(self, chunk: Chunk, n: int, llm: LLMClient) -> list[Sample]:
        """Call the LLM and convert the summary response into a Sample object.

        - n is ignored; always produces at most 1 sample
        - Returns [] if LLM responds with {"insufficient": true}
        - Returns [] if supporting_quote is missing or empty
        """
        messages = summary_generation_prompt(chunk.text)
        response = llm.complete_json(messages)

        if response.get("insufficient"):
            return []

        summary = response.get("summary", "")
        key_points = response.get("key_points", [])
        supporting_quote = response.get("supporting_quote", "")

        if not supporting_quote:
            return []

        return [
            Sample(
                id=uuid4().hex,
                chunk_id=chunk.id,
                dataset_type=DatasetType.summary,
                payload={"passage": chunk.text, "summary": summary, "key_points": key_points},
                source_text=chunk.text,
                supporting_quote=supporting_quote,
                source_citation=Citation(title="", source_uri=""),
            )
        ]


# Register at import time
register_generator(SummaryGenerator())
