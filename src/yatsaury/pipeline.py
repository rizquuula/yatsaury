"""Orchestrator — end-to-end pipeline from URIs to exported records."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from yatsaury.exporters import hf as _hf_reg  # noqa: F401 — triggers registration
from yatsaury.exporters import jsonl as _jsonl_reg  # noqa: F401 — triggers registration
from yatsaury.exporters import review_csv as _csv_reg  # noqa: F401 — triggers registration
from yatsaury.exporters.base import get_exporter
from yatsaury.generators import instruction as _instr_reg  # noqa: F401 — triggers registration
from yatsaury.generators import qa as _qa_reg  # noqa: F401 — triggers registration
from yatsaury.generators import rag as _rag_gen_reg  # noqa: F401 — triggers registration
from yatsaury.generators import summary as _summary_reg  # noqa: F401 — triggers registration
from yatsaury.generators.base import get_generator
from yatsaury.llm.client import LLMClient
from yatsaury.models import Sample
from yatsaury.processing.chunk import chunk_document
from yatsaury.processing.clean import clean_text
from yatsaury.quality.dedup import dedup_samples
from yatsaury.quality.verify import verify_samples
from yatsaury.schemas import alpaca as _alpaca_reg  # noqa: F401 — triggers registration
from yatsaury.schemas import chatml as _chatml_reg  # noqa: F401 — triggers registration
from yatsaury.schemas import completion as _completion_reg  # noqa: F401 — triggers registration
from yatsaury.schemas import qa as _qa_schema_reg  # noqa: F401 — triggers registration
from yatsaury.schemas import rag as _rag_schema_reg  # noqa: F401 — triggers registration
from yatsaury.schemas import raw as _raw_reg  # noqa: F401 — triggers registration
from yatsaury.schemas import sharegpt as _sharegpt_reg  # noqa: F401 — triggers registration
from yatsaury.schemas.base import get_schema
from yatsaury.sources.base import resolve_loader

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    dataset_types: list[str] = field(default_factory=lambda: ["qa"])
    schema_names: list[str] = field(default_factory=lambda: ["chatml"])
    output_formats: list[str] = field(default_factory=lambda: ["jsonl"])
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    chunk_size: int = 512
    chunk_overlap: int = 64
    per_chunk: int = 3
    min_score: float = 0.0
    lang: str = "auto"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    paraphrases: int = 1
    difficulty: list[str] = field(default_factory=list)


class Orchestrator:
    """Runs the full generation pipeline from URIs to exported records."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

    def run(
        self,
        source_uris: list[str],
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> list[dict]:
        cfg = self._config
        llm = LLMClient(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
        )
        all_samples: list[Sample] = []
        total_uris = len(source_uris)

        for uri_idx, uri in enumerate(source_uris):
            if progress_cb:
                progress_cb(f"Loading {uri}", uri_idx / max(total_uris, 1))

            try:
                loader = resolve_loader(uri)
                doc = loader.load(uri)
                doc = doc.model_copy(update={"raw_text": clean_text(doc.raw_text)})
            except Exception:
                logger.exception("Failed to load URI: %s", uri)
                continue

            chunks = chunk_document(doc, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)

            for chunk_idx, chunk in enumerate(chunks):
                if progress_cb:
                    progress_cb(
                        f"Processing chunk {chunk_idx + 1}/{len(chunks)}",
                        (uri_idx + (chunk_idx + 1) / max(len(chunks), 1)) / max(total_uris, 1),
                    )

                for dtype in cfg.dataset_types:
                    try:
                        generator = get_generator(dtype)
                        samples = generator.generate(
                            chunk, cfg.per_chunk, llm,
                            paraphrases=cfg.paraphrases,
                            difficulty=cfg.difficulty if cfg.difficulty else None,
                        )
                    except Exception:
                        logger.exception(
                            "Chunk %s failed for dataset_type=%s — skipping", chunk.id, dtype
                        )
                        continue

                    all_samples.extend(samples)

        verified = verify_samples(
            all_samples,
            llm,
            min_score=cfg.min_score,
            use_judge=cfg.min_score > 0.0,
        )
        deduplicated = dedup_samples(verified)

        all_records: list[dict] = []
        for sample in deduplicated:
            for schema_name in cfg.schema_names:
                try:
                    adapter = get_schema(schema_name)
                    if not adapter.supports(sample.dataset_type.value):
                        logger.warning(
                            "Schema %r does not support dataset_type=%r — skipping",
                            schema_name,
                            sample.dataset_type.value,
                        )
                        continue
                    record = adapter.render(sample)
                    all_records.append(record)
                except Exception:
                    logger.exception(
                        "Schema %r failed for sample %s — skipping",
                        schema_name,
                        sample.id,
                    )

        # Export — one file per schema per format
        for fmt in cfg.output_formats:
            try:
                exporter_cls = get_exporter(fmt)
                exporter = exporter_cls()
                for schema_name in cfg.schema_names:
                    out_path = cfg.output_dir / f"{schema_name}.{fmt}"
                    exporter.export(all_records, out_path)
            except Exception:
                logger.exception("Export failed for format %r", fmt)

        if progress_cb:
            progress_cb("Done", 1.0)

        return all_records
