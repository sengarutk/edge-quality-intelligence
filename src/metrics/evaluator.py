"""Evaluation of recorded runtime decisions and aggregation of multi-seed ablation runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from src.audit_log import AuditLogDB
from src.metrics.significance import apply_holm_bonferroni_correction, paired_significance_test
from src.metrics.stats import bootstrap_ci
from src.metrics.stream import compute_stream_metrics

# Metrics aggregated across seeds (keys produced by compute_stream_metrics).
AGGREGATED_METRICS = (
    "alerts_per_hour",
    "false_alerts_per_hour",
    "false_high_alerts_per_hour",
    "re_alerts_per_episode",
    "alerts_per_episode",
    "routing_recall",
    "mean_delay_frames",
    "non_normal_fraction",
    "latency_p95_ms",
)
# Metrics compared against BASELINE with paired tests (lower is better for all three).
TESTED_METRICS = ("alerts_per_hour", "false_alerts_per_hour", "false_high_alerts_per_hour", "re_alerts_per_episode")
MIN_PAIRS_FOR_TESTING = 6  # below this, a two-sided Wilcoxon test cannot reach p < 0.05


class BenchmarkEvaluator:
    """Computes operational metrics from decisions stored in an AuditLogDB."""

    def __init__(self, audit_db: Optional[AuditLogDB] = None, db_path: str = "data/audit_log.db") -> None:
        self.audit_db = audit_db or AuditLogDB(db_path=db_path)

    def decision_records(self) -> List[Dict[str, Any]]:
        """Decisions in emission order with the fields needed by compute_stream_metrics."""
        records = []
        for e in self.audit_db.query_events_in_order():
            payload = json.loads(e["raw_payload"]) if e.get("raw_payload") else {}
            records.append({
                "risk_state": e["risk_state"],
                "is_new_alert": bool(e.get("is_new_alert")),
                "latency_ms": payload.get("latency_ms"),
            })
        return records

    def compute_metrics(
        self,
        ground_truth_defect_steps: Sequence[int] = (),
        frame_interval_ms: float = 1000.0 / 30.0,
        deadline_ms: float = 1000.0 / 30.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        records = self.decision_records()
        if not records:
            return {"total_steps": 0}
        return compute_stream_metrics(
            records, ground_truth_defect_steps, frame_interval_ms=frame_interval_ms, deadline_ms=deadline_ms, **kwargs
        )

    def export_json(self, output_path: str | Path, **kwargs: Any) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.compute_metrics(**kwargs), indent=2), encoding="utf-8")


def _summarize(values: List[float], ci_level: float) -> Dict[str, Any]:
    res = bootstrap_ci(values, ci=ci_level, n_boot=2000, unit="seed")
    return {
        "mean": res["mean"],
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": res["median"],
        "ci_lower": res["ci_lower"],
        "ci_upper": res["ci_upper"],
        "ci_level": ci_level,
        "n": len(values),
        "values": values,
    }


def aggregate_ablation_results(results_dir: str | Path = "results/ablation", ci_level: float = 0.95) -> Dict[str, Any]:
    """Aggregate per-seed JSON files ``<scenario>/<POLICY>_seed<k>.json``.

    Returns ``{"scenarios": {scenario: {policy: stats}}}``. Undefined metrics (``None``)
    are excluded from the statistics instead of being replaced by a number.
    """
    res_path = Path(results_dir)
    scenarios: Dict[str, Any] = {}
    if not res_path.exists():
        return {"scenarios": scenarios}

    for s_dir in sorted(d for d in res_path.iterdir() if d.is_dir()):
        per_policy: Dict[str, List[Dict[str, Any]]] = {}
        for f in sorted(s_dir.glob("*_seed*.json")):
            policy = f.stem.rsplit("_seed", 1)[0]
            payload = json.loads(f.read_text(encoding="utf-8"))
            per_policy.setdefault(policy, []).append({"seed": payload["seed"], **payload["metrics"]})
        if not per_policy:
            continue

        block: Dict[str, Any] = {}
        for policy, runs in per_policy.items():
            runs.sort(key=lambda r: r["seed"])
            stats: Dict[str, Any] = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs]}
            for k in AGGREGATED_METRICS:
                vals = [float(r[k]) for r in runs if r.get(k) is not None and not math.isnan(float(r[k]))]
                if vals:
                    stats[k] = _summarize(vals, ci_level)
            block[policy] = stats

        base = per_policy.get("BASELINE")
        if base:
            base_by_seed = {r["seed"]: r for r in base}
            for policy, runs in per_policy.items():
                if policy == "BASELINE":
                    continue
                tests = []
                for k in TESTED_METRICS:
                    pairs = [(r[k], base_by_seed[r["seed"]][k]) for r in runs
                             if r["seed"] in base_by_seed and r.get(k) is not None
                             and base_by_seed[r["seed"]].get(k) is not None]
                    if len(pairs) < MIN_PAIRS_FOR_TESTING:
                        continue
                    a, b = zip(*pairs)
                    tests.append(paired_significance_test(list(a), list(b), k, policy, "BASELINE", unit="seed"))
                if tests:
                    block[policy]["significance_vs_baseline"] = apply_holm_bonferroni_correction(tests).to_dict(
                        orient="records"
                    )
        scenarios[s_dir.name] = block

    return {"scenarios": scenarios}
