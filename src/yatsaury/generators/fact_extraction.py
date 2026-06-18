"""Fact extraction and per-fact Q&A generation."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from yatsaury.llm.client import LLMClient
from yatsaury.llm.prompts import fact_extraction_prompt, qa_from_fact_prompt
from yatsaury.models import Chunk, Citation, DatasetType, Sample


@dataclass
class Fact:
    id: str
    text: str
    source_quote: str


def extract_facts(chunk: Chunk, llm: LLMClient) -> list[Fact]:
    """Call LLM to extract atomic facts from chunk.text."""
    messages = fact_extraction_prompt(chunk.text)
    response = llm.complete_json(messages)

    if response.get("insufficient"):
        return []

    facts = []
    for item in response.get("facts", []):
        text = item.get("text", "")
        source_quote = item.get("source_quote", "")
        if not source_quote:
            continue
        facts.append(Fact(id=uuid4().hex, text=text, source_quote=source_quote))

    return facts


def qa_from_fact(fact: Fact, chunk: Chunk, llm: LLMClient, n: int = 1) -> list[Sample]:
    """Generate n Q&A pairs for a specific fact."""
    messages = qa_from_fact_prompt(fact.text, chunk.text, n=n)
    response = llm.complete_json(messages)

    samples = []
    for pair in response.get("pairs", []):
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        supporting_quote = pair.get("supporting_quote", "")
        if not supporting_quote:
            continue
        samples.append(
            Sample(
                id=uuid4().hex,
                chunk_id=chunk.id,
                dataset_type=DatasetType.qa,
                payload={"question": question, "answer": answer},
                source_text=chunk.text,
                supporting_quote=supporting_quote,
                source_citation=Citation(title="", source_uri=""),
                fact_id=fact.id,
            )
        )

    return samples
