"""Tests for HfExporter."""
from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk

from yatsaury.exporters.base import get_exporter
from yatsaury.exporters.hf import HfExporter


class TestHfExporter:
    def test_export_creates_directory(self, tmp_path: Path):
        """After export, out_path directory exists."""
        records = [{"question": "Q?", "answer": "A."}]
        out_path = tmp_path / "dataset"
        HfExporter().export(records, out_path)
        assert out_path.exists()

    def test_export_three_records_roundtrip(self, tmp_path: Path):
        """Export 3 dicts then reload; dataset has 3 rows."""
        records = [
            {"question": "Q1?", "answer": "A1."},
            {"question": "Q2?", "answer": "A2."},
            {"question": "Q3?", "answer": "A3."},
        ]
        out_path = tmp_path / "dataset"
        HfExporter().export(records, out_path)
        ds = load_from_disk(str(out_path))
        assert len(ds) == 3

    def test_export_column_names_match(self, tmp_path: Path):
        """Reloaded dataset has same column names as input dict keys."""
        records = [{"question": "Q?", "answer": "A.", "score": 0.9}]
        out_path = tmp_path / "dataset"
        HfExporter().export(records, out_path)
        ds = load_from_disk(str(out_path))
        assert set(ds.column_names) == {"question", "answer", "score"}

    def test_export_values_preserved(self, tmp_path: Path):
        """Values round-trip correctly."""
        records = [{"text": "hello", "num": 42}]
        out_path = tmp_path / "dataset"
        HfExporter().export(records, out_path)
        ds = load_from_disk(str(out_path))
        assert ds[0]["text"] == "hello"
        assert ds[0]["num"] == 42

    def test_export_empty_records(self, tmp_path: Path):
        """Empty records list produces a dataset at out_path (0 rows or graceful)."""
        out_path = tmp_path / "empty_dataset"
        HfExporter().export([], out_path)
        # out_path should exist
        assert out_path.exists()

    def test_export_creates_parent_dirs(self, tmp_path: Path):
        """Parent directories are created if missing."""
        out_path = tmp_path / "deep" / "nested" / "dataset"
        HfExporter().export([{"x": 1}], out_path)
        assert out_path.exists()

    def test_registered_as_hf(self):
        """HfExporter registered under 'hf' name."""
        cls = get_exporter("hf")
        assert cls is HfExporter
