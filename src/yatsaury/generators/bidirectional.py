"""Bidirectional / inverse Q&A generation."""
from __future__ import annotations

from uuid import uuid4

from yatsaury.llm.client import LLMClient
from yatsaury.llm.prompts import inverse_question_prompt
from yatsaury.models import DatasetType, Sample


def generate_inverse(sample: Sample, llm: LLMClient) -> Sample | None:
    """Given a QA sample, generate the inverse question."""
    question = sample.payload.get("question", "")
    answer = sample.payload.get("answer", "")

    messages = inverse_question_prompt(question, answer)
    response = llm.complete_json(messages)

    inv_question = response.get("question", "")
    inv_answer = response.get("answer", "")
    supporting_quote = response.get("supporting_quote", "")

    if not supporting_quote:
        return None

    return Sample(
        id=uuid4().hex,
        chunk_id=sample.chunk_id,
        dataset_type=DatasetType.qa,
        payload={"question": inv_question, "answer": inv_answer},
        source_text=sample.source_text,
        supporting_quote=supporting_quote,
        source_citation=sample.source_citation,
        fact_id=sample.fact_id,
    )


def add_inverses(samples: list[Sample], llm: LLMClient) -> list[Sample]:
    """For each QA sample, optionally add its inverse. Returns original + inverses."""
    result: list[Sample] = []
    for sample in samples:
        result.append(sample)
        inverse = generate_inverse(sample, llm)
        if inverse is not None:
            result.append(inverse)
    return result
