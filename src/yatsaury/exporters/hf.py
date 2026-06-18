"""HuggingFace Dataset exporter."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets import Dataset

from yatsaury.exporters.base import register_exporter


class HfExporter:
    """Export records as a HuggingFace Dataset saved to disk."""

    def export(self, records: Iterable[dict], out_path: Path) -> None:
        """Save records as a HuggingFace Dataset to out_path directory.

        Creates parent dirs if needed.
        If records is empty, saves an empty dataset (0 rows).
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records_list = list(records)
        ds = Dataset.from_list(records_list)
        ds.save_to_disk(str(out_path))


# Register at import time
register_exporter("hf", HfExporter)
