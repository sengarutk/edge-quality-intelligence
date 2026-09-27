"""Tests for the stream metrics (src/metrics/stream.py) and the audit-backed evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.audit_log import AuditLogDB
from src.metrics.evaluator import BenchmarkEvaluator, aggregate_ablation_results
from src.metrics.stream import compute_stream_metrics, episodes_from_steps
from src.policy import TemporalPolicyEngine
from tests.helpers import inf, reading

HOURS_PER_STEP = 1 / 30 / 3600


def rec(risk: str = "NORMAL", alert: bool = False, lat=None):
    return {"risk_state": risk, "is_new_alert": alert, "latency_ms": lat}


def test_episodes_group_consecutive_steps():
    eps = episodes_from_steps([5, 6, 7, 20, 22, 21])
    assert [(e.start, e.end) for e in eps] == [(5, 8), (20, 23)]


def test_hand_computed_stream():
    # 100 steps, one episode at [40, 50); grace 15 covers [40, 65).
    records = [rec() for _ in range(100)]
    records[10] = rec("REVIEW_REQUIRED", True)        # false alert
    records[43] = rec("HIGH_SEVERITY", True)          # first alert of the episode, delay 3
    records[44] = rec("HIGH_SEVERITY", False)         # aggregated
    records[60] = rec("REVIEW_REQUIRED", True)        # re-alert inside the grace window
    records[80] = rec("HIGH_SEVERITY", True)          # false critical escalation
    m = compute_stream_metrics(records, range(40, 50))
    nominal_hours = (100 - 25) * HOURS_PER_STEP
    assert m["n_episodes"] == 1 and m["alerts"] == 4
    assert m["false_alerts"] == 2 and m["false_high_alerts"] == 1
    assert m["false_alerts_per_hour"] == pytest.approx(2 / nominal_hours)
    assert m["re_alerts"] == 1 and m["re_alerts_per_episode"] == 1.0
    assert m["routing_recall"] == 1.0 and m["mean_delay_frames"] == 3.0
    assert m["alerts_per_hour"] == pytest.approx(4 / (100 * HOURS_PER_STEP))


def test_missed_episode_counts_against_recall():
    records = [rec() for _ in range(200)]
    records[150] = rec("REVIEW_REQUIRED", True)  # 50 frames after onset: outside max_delay
    m = compute_stream_metrics(records, range(100, 170))
    assert m["routing_recall"] == 0.0 and m["mean_delay_frames"] is None


def test_no_ground_truth_leaves_recall_undefined():
    m = compute_stream_metrics([rec() for _ in range(10)], [])
    assert m["routing_recall"] is None and m["re_alerts_per_episode"] is None
    assert m["false_alerts_per_hour"] == 0.0


def test_latency_and_deadline():
    m = compute_stream_metrics([rec(lat=10.0)] * 95 + [rec(lat=50.0)] * 5, [], deadline_ms=33.3)
    assert m["deadline_miss_rate"] == pytest.approx(0.05)
    assert m["latency_max_ms"] == 50.0 and m["latency_p50_ms"] == 10.0


def test_empty_stream_is_rejected():
    with pytest.raises(ValueError):
        compute_stream_metrics([], [])


def test_evaluator_reads_decisions_in_emission_order(tmp_path: Path):
    db = AuditLogDB(db_path=str(tmp_path / "a.db"))
    e = TemporalPolicyEngine()
    for t in range(120):
        v, s = (0.95, 0.95) if 40 <= t < 80 else (0.05, 0.05)
        d = e.evaluate(inf(v), reading(s))
        d.timestamp_utc = "2026-01-01T00:00:00.000Z"  # identical timestamps: order must come from sequence_id
        db.insert_risk_event(d)
    m = BenchmarkEvaluator(audit_db=db).compute_metrics(ground_truth_defect_steps=range(40, 80))
    assert m["total_steps"] == 120 and m["routing_recall"] == 1.0 and m["false_alerts"] == 0
    # One incident: a REVIEW alert when vision confirms, then one upgrade to HIGH when the
    # slower sensor filter corroborates.
    assert m["alerts"] == 2 and m["re_alerts"] == 1
    db.close()


def test_aggregation_over_seeds(tmp_path: Path):
    for policy, fa in (("BASELINE", [100.0, 120.0, 110.0]), ("FULL_POLICY", [5.0, 7.0, 6.0])):
        for seed, v in zip((1, 2, 3), fa):
            out = tmp_path / "wl" / f"{policy}_seed{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"seed": seed, "metrics": {"false_alerts_per_hour": v, "routing_recall": None}}))
    summary = aggregate_ablation_results(tmp_path)
    full = summary["scenarios"]["wl"]["FULL_POLICY"]
    assert full["false_alerts_per_hour"]["mean"] == pytest.approx(6.0)
    assert full["false_alerts_per_hour"]["n"] == 3
    assert "routing_recall" not in full  # undefined values are dropped, not replaced by a number
    assert "significance_vs_baseline" not in full  # 3 pairs are too few for a Wilcoxon test
