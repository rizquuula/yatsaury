"""RAG passage generator (no LLM call)."""
from __future__ import annotations

from uuid import uuid4

from yatsaury.generators.base import register_generator
from yatsaury.llm.client import LLMClient
from yatsaury.models import Chunk, Citation, DatasetType, Sample


class RagGenerator:
    """Wrap a raw chunk as a RAG-ready Sample — no LLM call required."""

    dataset_type = "rag"

    def generate(self, chunk: Chunk, n: int, llm: LLMClient) -> list[Sample]:
        """Return exactly 1 Sample per chunk, fully grounded and verified.

        - n is ignored; always produces exactly 1 sample
        - No LLM call is made
        - grounding_score = 1.0, verified = True
        """
        return [
            Sample(
                id=uuid4().hex,
                chunk_id=chunk.id,
                dataset_type=DatasetType.rag,
                payload={
                    "text": chunk.text,
                    "title": "",
                    "page": chunk.page,
                    "char_span": list(chunk.char_span),
                },
                source_text=chunk.text,
                supporting_quote=chunk.text[:200],
                source_citation=Citation(title="", source_uri=""),
                grounding_score=1.0,
                verified=True,
            )
        ]


# Register at import time
register_generator(RagGenerator())
