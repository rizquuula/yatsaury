"""Tests for Orchestrator pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.pipeline import Orchestrator, OrchestratorConfig


def make_sample() -> Sample:
    return Sample(
        id=uuid4().hex,
        chunk_id="chk_test_0000",
        dataset_type=DatasetType.qa,
        payload={"question": "Q?", "answer": "A."},
        source_text="source",
        supporting_quote="source",
        source_citation=Citation(title="", source_uri=""),
    )


class TestOrchestratorRun:
    def test_run_returns_rendered_records(self, tmp_path: Path):
        """run() with a mocked generator returns rendered dicts."""
        config = OrchestratorConfig(
            dataset_types=["qa"],
            schema_names=["chatml"],
            output_formats=["jsonl"],
            output_dir=tmp_path / "out",
        )
        orch = Orchestrator(config)

        mock_sample = make_sample()

        with (
            patch("yatsaury.pipeline.get_generator") as mock_get_gen,
            patch("yatsaury.pipeline.LLMClient"),
        ):
            mock_gen = MagicMock()
            mock_gen.generate.return_value = [mock_sample]
            mock_get_gen.return_value = mock_gen

            records = orch.run(["hello world"])

        assert len(records) == 1
        assert "messages" in records[0]  # chatml renders to messages

    def test_run_skips_failing_chunk(self, tmp_path: Path):
        """A chunk that raises in generate() is skipped; run continues."""
        good_sample = make_sample()

        call_count = 0

        def side_effect(chunk, n, llm, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM timeout")
            return [good_sample]

        with (
            patch("yatsaury.pipeline.get_generator") as mock_get_gen,
            patch("yatsaury.pipeline.LLMClient"),
        ):
            mock_gen = MagicMock()
            mock_gen.generate.side_effect = side_effect
            mock_get_gen.return_value = mock_gen

            # Use text that will produce at least 2 chunks at small chunk_size
            long_text = "word " * 300
            config2 = OrchestratorConfig(
                dataset_types=["qa"],
                schema_names=["chatml"],
                output_formats=["jsonl"],
                output_dir=tmp_path / "out2",
                chunk_size=50,
                chunk_overlap=5,
            )
            orch2 = Orchestrator(config2)
            records = orch2.run([long_text])

        # At least one record despite the first chunk failing
        assert len(records) >= 1

    def test_output_file_written(self, tmp_path: Path):
        """run() writes a JSONL output file."""
        config = OrchestratorConfig(
            dataset_types=["qa"],
            schema_names=["chatml"],
            output_formats=["jsonl"],
            output_dir=tmp_path / "out",
        )
        orch = Orchestrator(config)

        with (
            patch("yatsaury.pipeline.get_generator") as mock_get_gen,
            patch("yatsaury.pipeline.LLMClient"),
        ):
            mock_gen = MagicMock()
            mock_gen.generate.return_value = [make_sample()]
            mock_get_gen.return_value = mock_gen

            orch.run(["hello world"])

        out_file = tmp_path / "out" / "chatml.jsonl"
        assert out_file.exists()

    def test_progress_cb_called(self, tmp_path: Path):
        """progress_cb is invoked at least once during run()."""
        config = OrchestratorConfig(
            dataset_types=["qa"],
            schema_names=["chatml"],
            output_formats=["jsonl"],
            output_dir=tmp_path / "out",
        )
        orch = Orchestrator(config)
        cb_calls: list[tuple[str, float]] = []

        with (
            patch("yatsaury.pipeline.get_generator") as mock_get_gen,
            patch("yatsaury.pipeline.LLMClient"),
        ):
            mock_gen = MagicMock()
            mock_gen.generate.return_value = [make_sample()]
            mock_get_gen.return_value = mock_gen

            orch.run(["hello world"], progress_cb=lambda msg, pct: cb_calls.append((msg, pct)))

        assert len(cb_calls) >= 1
