"""Fact coverage reporting."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from yatsaury.models import Sample


@dataclass
class FactCoverage:
    fact_id: str
    count: int
    samples: list[Sample] = field(default_factory=list)


def coverage_report(samples: list[Sample], min_count: int = 2) -> dict:
    """Group samples by fact_id and compute per-fact coverage."""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    no_fact_id_count = 0

    for sample in samples:
        if sample.fact_id is None:
            no_fact_id_count += 1
        else:
            grouped[sample.fact_id].append(sample)

    undercovered = []
    well_covered = []

    for fact_id, fact_samples in grouped.items():
        count = len(fact_samples)
        if count < min_count:
            undercovered.append({
                "fact_id": fact_id,
                "count": count,
                "sample_ids": [s.id for s in fact_samples],
            })
        else:
            well_covered.append({"fact_id": fact_id, "count": count})

    return {
        "total_facts": len(grouped),
        "total_samples": len(samples),
        "undercovered": undercovered,
        "well_covered": well_covered,
        "no_fact_id": no_fact_id_count,
    }
