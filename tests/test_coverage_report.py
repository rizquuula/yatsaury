# tests/test_coverage_report.py
from tests.helpers import make_sample
from yatsaury.quality.coverage import coverage_report


def test_coverage_report_basic():
    s1 = make_sample(fact_id="f1")
    s2 = make_sample(fact_id="f1")
    s3 = make_sample(fact_id="f2")
    report = coverage_report([s1, s2, s3], min_count=2)
    assert report["total_facts"] == 2
    assert report["total_samples"] == 3
    assert len(report["well_covered"]) == 1
    assert report["well_covered"][0]["fact_id"] == "f1"
    assert len(report["undercovered"]) == 1
    assert report["undercovered"][0]["count"] == 1


def test_coverage_report_no_fact_id():
    s = make_sample(fact_id=None)
    report = coverage_report([s], min_count=2)
    assert report["no_fact_id"] == 1
    assert report["total_facts"] == 0


def test_coverage_report_empty():
    report = coverage_report([], min_count=2)
    assert report["total_facts"] == 0
    assert report["total_samples"] == 0
    assert report["undercovered"] == []
    assert report["well_covered"] == []


def test_coverage_report_all_well_covered():
    samples = [make_sample(fact_id="f1"), make_sample(fact_id="f1"), make_sample(fact_id="f1")]
    report = coverage_report(samples, min_count=2)
    assert report["undercovered"] == []
    assert len(report["well_covered"]) == 1
    assert report["well_covered"][0]["count"] == 3


def test_coverage_report_min_count_respected():
    s1, s2 = make_sample(fact_id="f1"), make_sample(fact_id="f1")
    report_2 = coverage_report([s1, s2], min_count=2)
    report_3 = coverage_report([s1, s2], min_count=3)
    assert len(report_2["well_covered"]) == 1
    assert len(report_3["undercovered"]) == 1
