#!/usr/bin/env python3
"""Policy ablation: 8 policy variants + 3 external alarm baselines x 6 workloads x 21 units.

Each run replays one generated workload timeline (src/experiments/workloads.py) through
the TemporalPolicyEngine. Visual scores are sampled from the category's PatchCore score
bank; optical health is computed on real held-out frames (degraded when the timeline
says so) with the category's calibrated blur threshold; sensor readings come from the
SensorSimulator with the timeline's mechanical faults, dropouts and drift.

The MQTT/spool path is deliberately not exercised here; it is evaluated against a real
broker in scripts/benchmark_spooler_resilience.py.

An experimental unit is (category, replicate). Unit ids are 100 * r + c with replicate r in {0, 1, 2}
and category index c in 0..6; the unit id is the random seed of the timeline, the score draws and the
sensor noise. ``--rho`` > 0 replays the same timelines with temporally correlated visual scores
(src/experiments/runtime_sim.py) and writes to a separate directory.

Outputs: <out-dir>/<workload>/<POLICY>_seed<unit>.json and <out-dir>/ablation_summary.json
(default out-dir results/ablation, or results/ablation_rho<rho> when --rho > 0).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PolicyConfig, PolicyMode, load_policy_config  # noqa: E402
from src.experiments.runtime_sim import simulate_workload  # noqa: E402
from src.experiments.workloads import ScoreBank, load_workload  # noqa: E402
from src.metrics.evaluator import aggregate_ablation_results  # noqa: E402
from src.metrics.stream import compute_stream_metrics  # noqa: E402

POLICY_MODES = [
    PolicyMode.BASELINE, PolicyMode.EMA_ONLY, PolicyMode.EMA_KOFN, PolicyMode.NO_COOLDOWN,
    PolicyMode.NO_FUSION, PolicyMode.NO_DIVERGENCE, PolicyMode.NO_STATE_GATING, PolicyMode.FULL_POLICY,
    PolicyMode.DELAY_TIMER, PolicyMode.EMA_HYSTERESIS, PolicyMode.DECISION_FUSION,
]
REPLICATES = [0, 1, 2]


def run_one(workload_path: str, bank_path: str, mode: str, replicate: int, unit: int, out_dir: str,
            rho: float = 0.0) -> Dict[str, Any]:
    logger.remove()
    workload = load_workload(workload_path)
    bank = ScoreBank(bank_path)
    policy_cfg: PolicyConfig = load_policy_config().model_copy(deep=True)
    policy_cfg.policy_mode = PolicyMode(mode)
    records, tl = simulate_workload(workload, bank, policy_cfg, unit, rho=rho)

    metrics = compute_stream_metrics(records, tl.defect_steps())
    reasons: Dict[str, int] = {}
    for r in records:
        if r["is_new_alert"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    payload = {
        "workload": workload.name, "policy_mode": mode, "seed": unit, "replicate": replicate, "rho": rho,
        "category": bank.category, "total_steps": tl.n, "metrics": metrics,
        "alert_steps": [i for i, r in enumerate(records) if r["is_new_alert"]],
        "alert_reasons": reasons,
    }
    out = Path(out_dir) / workload.name / f"{mode}_seed{unit}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", default=str(PROJECT_ROOT / "configs" / "scenarios"))
    ap.add_argument("--score-bank", default=str(PROJECT_ROOT / "results" / "score_bank"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.0, help="temporal correlation of visual scores (0 = i.i.d.)")
    args = ap.parse_args()

    default = "ablation" if args.rho == 0 else f"ablation_rho{args.rho:g}"
    out_dir = Path(args.out_dir or PROJECT_ROOT / "results" / default)
    for old in out_dir.glob("*/*_seed*.json"):
        old.unlink()
    banks = sorted(Path(args.score_bank).glob("*.npz"))
    if not banks:
        raise SystemExit("No score banks found; run scripts/build_score_bank.py first.")
    workloads = sorted(Path(args.workloads).glob("*.yaml"))

    tasks = []
    for w in workloads:
        for rep in REPLICATES:
            for bi, bank in enumerate(banks):
                unit = 100 * rep + bi  # one experimental unit = (replicate, category)
                for mode in POLICY_MODES:
                    tasks.append((str(w), str(bank), mode.value, rep, unit, str(out_dir), args.rho))
    print(f"Running {len(tasks)} runs ({len(workloads)} workloads x {len(POLICY_MODES)} policies x "
          f"{len(REPLICATES) * len(banks)} units, rho={args.rho:g})")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in concurrent.futures.as_completed([ex.submit(run_one, *t) for t in tasks]):
            f.result()

    summary = aggregate_ablation_results(out_dir)
    summary["design"] = {
        "policy_modes": [m.value for m in POLICY_MODES], "workloads": [w.stem for w in workloads],
        "replicates": REPLICATES, "unit_id": "100 * replicate + category_index (also the random seed)",
        "categories": [b.stem for b in banks], "units_per_cell": len(REPLICATES) * len(banks),
        "steps_per_run": 9000, "frame_rate_hz": 30, "rho": args.rho,
    }
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"Wrote {out_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
