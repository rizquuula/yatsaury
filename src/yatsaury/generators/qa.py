"""QA pair generator."""
from __future__ import annotations

from uuid import uuid4

from yatsaury.generators.base import register_generator
from yatsaury.llm.client import LLMClient
from yatsaury.llm.prompts import paraphrase_prompt, qa_generation_prompt
from yatsaury.models import Chunk, Citation, DatasetType, Sample


class QaGenerator:
    """Generate Q&A pairs from a chunk using an LLM."""

    dataset_type = "qa"

    def generate(
        self,
        chunk: Chunk,
        n: int,
        llm: LLMClient,
        paraphrases: int = 1,
        difficulty: list[str] | None = None,
        **kwargs,
    ) -> list[Sample]:
        """Call the LLM and convert response pairs into Sample objects.

        - Returns [] if LLM responds with {"insufficient": true}
        - Drops pairs with empty supporting_quote
        - If paraphrases > 1, generates paraphrase variants sharing same fact_id
        - If difficulty is a list, makes one call per difficulty level and merges
        """
        difficulties: list[str | None] = difficulty if difficulty else [None]

        all_samples: list[Sample] = []
        for diff_level in difficulties:
            messages = qa_generation_prompt(chunk.text, n=n, difficulty=diff_level)
            response = llm.complete_json(messages)

            if response.get("insufficient"):
                continue

            pairs = response.get("pairs", [])
            for pair in pairs:
                question = pair.get("question", "")
                answer = pair.get("answer", "")
                supporting_quote = pair.get("supporting_quote", "")

                if not supporting_quote:
                    continue

                fact_id = uuid4().hex
                base_sample = Sample(
                    id=uuid4().hex,
                    chunk_id=chunk.id,
                    dataset_type=DatasetType.qa,
                    payload={"question": question, "answer": answer},
                    source_text=chunk.text,
                    supporting_quote=supporting_quote,
                    source_citation=Citation(title="", source_uri=""),
                    fact_id=fact_id,
                )
                all_samples.append(base_sample)

                if paraphrases > 1:
                    para_messages = paraphrase_prompt(question, answer, n=paraphrases - 1)
                    para_response = llm.complete_json(para_messages)
                    variants = para_response.get("variants", [])
                    for variant in variants:
                        v_question = variant.get("question", "")
                        v_answer = variant.get("answer", "")
                        v_quote = variant.get("supporting_quote", "")
                        if not v_quote:
                            continue
                        all_samples.append(
                            Sample(
                                id=uuid4().hex,
                                chunk_id=chunk.id,
                                dataset_type=DatasetType.qa,
                                payload={"question": v_question, "answer": v_answer},
                                source_text=chunk.text,
                                supporting_quote=v_quote,
                                source_citation=Citation(title="", source_uri=""),
                                fact_id=fact_id,
                            )
                        )

        return all_samples


# Register at import time
register_generator(QaGenerator())
