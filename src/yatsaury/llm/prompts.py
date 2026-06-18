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


SYSTEM_PARAPHRASE = """\
You are a precise dataset annotator. Your task is to generate paraphrase variants of a Q&A pair.

Rules:
1. Each variant must convey the same fact but use different wording.
2. Each variant MUST include a supporting_quote verbatim from the original text.
3. Respond ONLY with valid JSON matching this schema:
   {"variants": [{"question": "...", "answer": "...", "supporting_quote": "..."}]}
"""


def paraphrase_prompt(question: str, answer: str, n: int) -> list[dict]:
    """Prompt to generate N paraphrase variants of a Q&A pair.

    Expected JSON: {"variants": [{"question": "...", "answer": "...", "supporting_quote": "..."}]}
    """
    user_content = (
        f"Generate {n} differently-worded paraphrase variants of this Q&A pair.\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    return [
        {"role": "system", "content": SYSTEM_PARAPHRASE},
        {"role": "user", "content": user_content},
    ]


SYSTEM_FACT_EXTRACTION = """\
You are a precise fact extractor. Extract atomic facts from the provided text.

Rules:
1. Each fact must be a single, self-contained statement directly from the text.
2. Each fact MUST include a source_quote — a verbatim substring of the provided text.
3. If the text does not contain extractable facts, respond with: {"insufficient": true}
4. Respond ONLY with valid JSON matching this schema:
   {"facts": [{"text": "...", "source_quote": "..."}]}
"""

SYSTEM_QA_FROM_FACT = """\
You are a precise dataset annotator. Generate Q&A pairs about a specific fact.

Rules:
1. Each Q&A must directly test knowledge of the given fact.
2. Each pair MUST include a supporting_quote — a verbatim substring of the source text.
3. Respond ONLY with valid JSON matching this schema:
   {"pairs": [{"question": "...", "answer": "...", "supporting_quote": "..."}]}
"""

SYSTEM_INVERSE_QUESTION = """\
You are a precise dataset annotator. Generate an inverse Q&A pair.

Rules:
1. Given a Q&A pair, create an inverse where the answer IS the original question's topic.
2. MUST include a supporting_quote verbatim from the original answer.
3. Respond ONLY with valid JSON:
   {"question": "...", "answer": "...", "supporting_quote": "..."}
"""


def fact_extraction_prompt(chunk_text: str) -> list[dict]:
    """Prompt to extract atomic facts from a chunk."""
    return [
        {"role": "system", "content": SYSTEM_FACT_EXTRACTION},
        {"role": "user", "content": f"Extract atomic facts from this text:\n\n{chunk_text}"},
    ]


def qa_from_fact_prompt(fact_text: str, chunk_text: str, n: int = 1) -> list[dict]:
    """Prompt to generate Q&A pairs for a specific fact."""
    return [
        {"role": "system", "content": SYSTEM_QA_FROM_FACT},
        {
            "role": "user",
            "content": (
                f"Generate {n} Q&A pair(s) about this fact: {fact_text}\n\n"
                f"Source text:\n{chunk_text}"
            ),
        },
    ]


def inverse_question_prompt(question: str, answer: str) -> list[dict]:
    """Prompt to generate an inverse Q&A pair."""
    return [
        {"role": "system", "content": SYSTEM_INVERSE_QUESTION},
        {
            "role": "user",
            "content": (
                "Generate an inverse question where the answer IS"
                " the original question's subject.\n\n"
                f"Original Q: {question}\nOriginal A: {answer}"
            ),
        },
    ]
