"""Grounding-first prompt templates for Yatsaury."""
from __future__ import annotations

SYSTEM_INSTRUCTION = """\
You are a precise dataset annotator. Your task is to generate instruction/input/output triples \
strictly grounded in the provided text.

Rules:
1. Every output MUST be directly supported by the text. Do not infer or add outside knowledge.
2. Each triple MUST include a supporting_quote that is a verbatim substring of the provided text.
3. If the text does not contain enough information, respond with: {"insufficient": true}
4. Respond ONLY with valid JSON matching this schema:
   {"triples": [{"instruction": "...", "input": "", "output": "...", "supporting_quote": "..."}]}
"""

SYSTEM_SUMMARY = """\
You are a precise dataset annotator. Your task is to summarize the provided text.

Rules:
1. The summary MUST be grounded in the text. Do not add outside knowledge.
2. Include a supporting_quote that is a verbatim substring of the provided text.
3. If the text is not suitable for summarization, respond with: {"insufficient": true}
4. Respond ONLY with valid JSON matching this schema:
   {"summary": "...", "key_points": ["...", "..."], "supporting_quote": "..."}
"""

SYSTEM_QA = """\
You are a precise dataset annotator. Your task is to generate question-answer pairs \
strictly grounded in the provided text.

Rules:
1. Every answer MUST be directly supported by the text. Do not infer or add outside knowledge.
2. Each pair MUST include a supporting_quote that is a verbatim substring of the provided text.
3. If the text does not contain enough information to generate a meaningful Q&A pair, \
respond with: {"insufficient": true}
4. Respond ONLY with valid JSON matching this schema:
   {"pairs": [{"question": "...", "answer": "...", "supporting_quote": "..."}]}
"""


def qa_generation_prompt(
    chunk_text: str,
    n: int = 3,
    lang: str = "auto",
    difficulty: str | None = None,
) -> list[dict]:
    """Return messages list for JSON-mode Q&A generation.

    The system prompt instructs the model to ground answers in the chunk
    and include a verbatim supporting_quote.
    """
    lang_instruction = "" if lang == "auto" else f" Respond in language: {lang}."
    difficulty_instruction = (
        "" if difficulty is None else f" Target difficulty level: {difficulty}."
    )
    user_content = (
        f"Generate {n} question-answer pairs from the following text."
        f"{lang_instruction}{difficulty_instruction}\n\n"
        f"Text:\n{chunk_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_QA},
        {"role": "user", "content": user_content},
    ]


def instruction_generation_prompt(
    chunk_text: str,
    n: int = 3,
    lang: str = "auto",
) -> list[dict]:
    """Return messages list for JSON-mode instruction/input/output triple generation."""
    lang_instruction = "" if lang == "auto" else f" Respond in language: {lang}."
    user_content = (
        f"Generate {n} instruction-output triples from the following text."
        f"{lang_instruction}\n\nText:\n{chunk_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def summary_generation_prompt(
    chunk_text: str,
    lang: str = "auto",
) -> list[dict]:
    """Return messages list for JSON-mode summary generation."""
    lang_instruction = "" if lang == "auto" else f" Respond in language: {lang}."
    user_content = (
        f"Summarize the following text.{lang_instruction}\n\nText:\n{chunk_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_SUMMARY},
        {"role": "user", "content": user_content},
    ]
