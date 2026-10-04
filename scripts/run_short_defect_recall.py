#!/usr/bin/env python3
"""Recall as a function of defect duration (how short a defect each policy can catch).

For each duration L (frames) a 9,000-frame workload contains 20 defect episodes of exactly L
frames on otherwise good parts. Visual scores come from the PatchCore score banks (defect pool
inside episodes, good pool elsewhere); sensors stay nominal, so detection is purely visual.
Recall uses the same rule as the paper (a non-normal decision within 15 frames of onset).
Units, seeds and score banks are those of the main ablation (7 categories x 3 seeds).

Output: results/short_defect_recall.json (separate from results/ablation/, whose structure is
parsed by the companion paper and must not change).
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.config import PolicyMode, load_policy_config  # noqa: E402
from src.experiments.runtime_sim import simulate_workload  # noqa: E402
from src.experiments.workloads import EventFamily, ScoreBank, WorkloadConfig  # noqa: E402
from src.metrics.evaluator import _summarize  # noqa: E402
from src.metrics.stream import compute_stream_metrics  # noqa: E402

DURATIONS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
POLICIES = ["BASELINE", "EMA_ONLY", "EMA_KOFN", "FULL_POLICY"]
SEEDS = [11, 23, 37]
RECALL_TARGET = 0.95


def run(length: int, policy: str, bank_path: str, unit: int) -> Dict:
    logger.remove()
    wl = WorkloadConfig(name=f"short_{length}", description="short defects", total_steps=9000,
                        events=[EventFamily(kind="vision_defect", count=20, min_len=length, max_len=length)])
    cfg = load_policy_config().model_copy(deep=True)
    cfg.policy_mode = PolicyMode(policy)
    records, tl = simulate_workload(wl, ScoreBank(bank_path), cfg, unit)
    m = compute_stream_metrics(records, tl.defect_steps())
    return {"length": length, "policy": policy, "unit": unit, "recall": m["routing_recall"],
            "delay": m["mean_delay_frames"]}


def main() -> None:
    banks = sorted((PROJECT_ROOT / "results" / "score_bank").glob("*.npz"))
    tasks = [(L, p, str(b), 100 * si + bi) for L in DURATIONS for p in POLICIES
             for si, _ in enumerate(SEEDS) for bi, b in enumerate(banks)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(run, *zip(*tasks), chunksize=4))

    k = load_policy_config().confirmation_window
    out: Dict = {"durations_frames": DURATIONS, "policies": {}, "recall_target": RECALL_TARGET,
                 "k_of_n": [k.required_k, k.window_size_n], "units": len(SEEDS) * len(banks)}
    for p in POLICIES:
        per_len: List[Dict] = []
        for L in DURATIONS:
            vals = [r["recall"] for r in rows if r["policy"] == p and r["length"] == L]
            per_len.append({"length": L, "recall": _summarize(vals, 0.95)})
        reached = [e["length"] for e in per_len if e["recall"]["mean"] >= RECALL_TARGET]
        out["policies"][p] = {"per_length": per_len,
                              "min_length_for_target_recall": min(reached) if reached else None}
    dest = PROJECT_ROOT / "results" / "short_defect_recall.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for p in POLICIES:
        print(p, [round(e["recall"]["mean"], 2) for e in out["policies"][p]["per_length"]],
              "min L:", out["policies"][p]["min_length_for_target_recall"])


if __name__ == "__main__":
    main()
