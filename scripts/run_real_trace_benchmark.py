#!/usr/bin/env python3
"""Replay of two synthetic run-to-failure sensor traces through every policy mode.

The traces (src/runtime/trace_replay.py) are hand-parameterized curves whose shape is
inspired by the NASA IMS bearing and C-MAPSS turbofan datasets; they are NOT derived
from those datasets. The camera sees good parts throughout, so visual scores are drawn
from the nominal PatchCore score pools of the MVTec categories (one replay per category).

Ground truth: steps at and after the fault onset are actionable. The first
``CALIB_STEPS`` samples calibrate the sensor baseline and are excluded from scoring.
All policies are scored with the same code (src/metrics/stream.py).
Output: results/real_trace_benchmark_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.config import PolicyMode, load_policy_config  # noqa: E402
from src.experiments.workloads import ScoreBank  # noqa: E402
from src.inference_service import InferenceResult, OpticalHealthStatus  # noqa: E402
from src.metrics.evaluator import _summarize  # noqa: E402
from src.metrics.stream import compute_stream_metrics  # noqa: E402
from src.runtime.alarm_baselines import make_engine  # noqa: E402
from src.trace_replay import RealSensorTraceReplay, generate_cmapss_turbofan_trace, generate_ims_bearing_trace  # noqa: E402

CALIB_STEPS = 60
TRACES = {
    "bearing_proxy": (generate_ims_bearing_trace, 420),
    "thermal_creep_proxy": (generate_cmapss_turbofan_trace, 390),
}
METRICS = ("false_alerts_per_hour", "alerts_per_hour", "routing_recall", "mean_delay_frames",
           "first_flag_rel_onset", "alerts_before_onset", "alerts_from_onset", "high_alerts_from_onset")


def replay(trace_path: Path, onset: int, mode: PolicyMode, bank: ScoreBank, seed: int) -> dict:
    replay_src = RealSensorTraceReplay(trace_path, calibration_window_steps=CALIB_STEPS)
    cfg = load_policy_config().model_copy(deep=True)
    cfg.policy_mode = mode
    engine = make_engine(cfg)
    rng = np.random.default_rng(seed)
    health = OpticalHealthStatus(is_valid=True, laplacian_var=10 * bank.blur_threshold, mean_brightness=120.0)
    records = []
    for step, reading in enumerate(replay_src):
        inf = InferenceResult(timestamp_utc=reading.timestamp_utc, camera_id="cam",
                              vision_score=float(rng.choice(bank.pools[0])), is_blurred=False,
                              is_occluded=False, optical_health=health, latency_ms=0.0)
        d = engine.evaluate(inf, reading)
        if step >= CALIB_STEPS:
            records.append({"risk_state": d.risk_state.value, "is_new_alert": d.is_new_alert})
    n = len(records)
    m = compute_stream_metrics(records, range(onset - CALIB_STEPS, n))
    # Recall and delay count any non-normal decision, including an incident that was already open
    # before the onset. These fields make explicit when the policy first flagged the trace and how
    # many alerts (entries into the operator queue) fell before and after the labelled onset.
    on = onset - CALIB_STEPS
    flagged = [i for i, r in enumerate(records) if r["risk_state"] != "NORMAL"]
    m["first_flag_rel_onset"] = (flagged[0] - on) if flagged else None
    m["alerts_before_onset"] = sum(1 for r in records[:on] if r["is_new_alert"])
    m["alerts_from_onset"] = sum(1 for r in records[on:] if r["is_new_alert"])
    m["high_alerts_from_onset"] = sum(1 for r in records[on:] if r["is_new_alert"] and r["risk_state"] == "HIGH_SEVERITY")
    return m


def main() -> None:
    logger.remove()
    banks = [ScoreBank(p) for p in sorted((PROJECT_ROOT / "results" / "score_bank").glob("*.npz"))]
    out = {"calibration_steps_excluded": CALIB_STEPS, "units": [b.category for b in banks], "traces": {}}
    for name, (gen, onset) in TRACES.items():
        path = gen(PROJECT_ROOT / "data" / "traces" / f"{name}.csv")
        per_mode = {}
        for mode in PolicyMode:  # our variants and the external baselines
            runs = [replay(path, onset, mode, b, seed=i) for i, b in enumerate(banks)]
            per_mode[mode.value] = {k: _summarize([r[k] for r in runs if r.get(k) is not None], 0.95)
                                    for k in METRICS if any(r.get(k) is not None for r in runs)}
        out["traces"][name] = {"onset_step": onset, "total_steps": 600, "policies": per_mode}
    dest = PROJECT_ROOT / "results" / "real_trace_benchmark_summary.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, t in out["traces"].items():
        for m in ("BASELINE", "FULL_POLICY"):
            s = t["policies"][m]
            print(name, m, {k: round(v["mean"], 3) for k, v in s.items()})


if __name__ == "__main__":
    main()
