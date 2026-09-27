#!/usr/bin/env python3
"""One-at-a-time sensitivity of FULL_POLICY to its main parameters.

Each parameter is varied around the fixed defaults of configs/policy_config.yaml while the
others stay at their defaults. The same workloads and experimental units as the ablation
are used (7 MVTec categories x 3 seeds). This sweep describes how results move with the
parameters; the defaults reported in the paper were fixed before any sweep was run and
are not re-tuned on these results.
Output: results/sensitivity/sensitivity_summary.json
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.config import PolicyMode, load_policy_config  # noqa: E402
from src.experiments.runtime_sim import simulate_workload  # noqa: E402
from src.experiments.workloads import ScoreBank, load_workload  # noqa: E402
from src.metrics.evaluator import _summarize  # noqa: E402
from src.metrics.stream import compute_stream_metrics  # noqa: E402

WORKLOADS = ("transient_glitches", "sustained_defects", "multimodal_faults")
SEEDS = [11, 23, 37]
SWEEPS: Dict[str, List[float]] = {
    "required_k": [2, 3, 4, 5, 6],
    "vision_high": [0.7, 0.75, 0.8, 0.85, 0.9],
    "cross_modal_divergence": [0.3, 0.45, 0.6],
    "cooldown_steps": [5, 15, 45, 90],
    "alpha_vision": [0.2, 0.35, 0.5],
}
METRICS = ("alerts_per_hour", "false_alerts_per_hour", "false_high_alerts_per_hour", "routing_recall", "mean_delay_frames")


def apply(param: str, value: float):
    cfg = load_policy_config().model_copy(deep=True)
    cfg.policy_mode = PolicyMode.FULL_POLICY
    if param == "required_k":
        cfg.confirmation_window.required_k = int(value)
    elif param == "vision_high":
        cfg.thresholds.vision_high = float(value)
    elif param == "cross_modal_divergence":
        cfg.thresholds.cross_modal_divergence = float(value)
    elif param == "cooldown_steps":
        cfg.cooldown.cooldown_steps = int(value)
    elif param == "alpha_vision":
        cfg.temporal_smoothing.alpha_vision = float(value)
    else:
        raise KeyError(param)
    return cfg


def run(param: str, value: float, workload: str, bank_path: str, unit: int) -> Dict[str, Any]:
    logger.remove()
    records, tl = simulate_workload(
        load_workload(PROJECT_ROOT / "configs" / "scenarios" / f"{workload}.yaml"), ScoreBank(bank_path),
        apply(param, value), unit,
    )
    m = compute_stream_metrics(records, tl.defect_steps())
    return {"param": param, "value": value, "workload": workload, **{k: m.get(k) for k in METRICS}}


def main() -> None:
    banks = sorted((PROJECT_ROOT / "results" / "score_bank").glob("*.npz"))
    tasks = [(p, v, w, str(b), 100 * si + bi)
             for p, values in SWEEPS.items() for v in values for w in WORKLOADS
             for si, _ in enumerate(SEEDS) for bi, b in enumerate(banks)]
    print(f"{len(tasks)} runs")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(run, *zip(*tasks), chunksize=4))

    summary: Dict[str, Any] = {"defaults": load_policy_config().model_dump(mode="json"), "sweeps": {}}
    for p, values in SWEEPS.items():
        summary["sweeps"][p] = {}
        for w in WORKLOADS:
            summary["sweeps"][p][w] = []
            for v in values:
                sub = [r for r in rows if r["param"] == p and r["value"] == v and r["workload"] == w]
                entry = {"value": v}
                for k in METRICS:
                    vals = [r[k] for r in sub if r[k] is not None]
                    if vals:
                        entry[k] = _summarize(vals, 0.95)
                summary["sweeps"][p][w].append(entry)
    out = PROJECT_ROOT / "results" / "sensitivity" / "sensitivity_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for p in SWEEPS:
        for e in summary["sweeps"][p]["multimodal_faults"]:
            print(p, e["value"], {k: round(e[k]["mean"], 2) for k in METRICS if k in e})


if __name__ == "__main__":
    main()
