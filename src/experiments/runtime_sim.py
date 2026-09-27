"""Replay one workload timeline through the policy engine (shared by the ablation and sensitivity studies)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import OpticalHealthConfig, PolicyConfig, SystemConfig, load_sensor_config
from src.experiments.workloads import ScoreBank, Timeline, WorkloadConfig, build_timeline
from src.inference_service import InferenceEngine, InferenceResult
from src.policy import TemporalPolicyEngine
from src.sensor_simulator import MachineState, SensorSimulator


def simulate_workload(
    workload: WorkloadConfig, bank: ScoreBank, policy_cfg: PolicyConfig, unit: int
) -> Tuple[List[Dict[str, Any]], Timeline]:
    """Run ``workload`` for experimental unit ``unit`` and return per-step decision records.

    The timeline, visual-score sampling and sensor noise depend only on ``unit``, so every
    policy configuration sees exactly the same inputs for a given unit (paired design).
    """
    tl = build_timeline(workload, seed=unit)
    rng = np.random.default_rng(10_000 + unit)
    engine = TemporalPolicyEngine(config=policy_cfg)

    sys_cfg = SystemConfig(optical_health=OpticalHealthConfig(blur_laplacian_threshold=bank.blur_threshold))
    health_engine = InferenceEngine(config=sys_cfg, seed=unit)
    health_cache: Dict[tuple, Any] = {}

    sensor = SensorSimulator(config=load_sensor_config().model_copy(deep=True), seed=unit)
    for _ in range(400):  # reach the running thermal equilibrium before the run starts
        sensor.step(MachineState.RUNNING)

    records: List[Dict[str, Any]] = []
    for t in range(tl.n):
        key = (t % len(bank.frames), int(tl.optical[t]))
        if key not in health_cache:
            health_cache[key] = health_engine.check_optical_health(bank.frame(*key))
        health = health_cache[key]
        # Draw from the pool every step so the random stream does not depend on optical state.
        drawn = float(rng.choice(bank.pools[int(tl.vision_source[t])]))
        score = drawn if health.is_valid else 0.0
        inf = InferenceResult(
            timestamp_utc="1970-01-01T00:00:00.000Z", camera_id="cam", vision_score=score,
            is_blurred=health.degradation_reason == "OPTICAL_BLURRED",
            is_occluded=health.degradation_reason in ("OPTICAL_OCCLUDED_DARK", "OPTICAL_OCCLUDED_BRIGHT"),
            optical_health=health, latency_ms=0.0,
        )
        reading = sensor.step(
            machine_state=MachineState(tl.machine_state[t]),
            inject_fault=bool(tl.mechanical[t]),
            simulate_dropout=tl.dropout[t] or None,
            temperature_offset_c=float(tl.drift_c[t]),
        )
        d = engine.evaluate(inf, reading)
        records.append({"risk_state": d.risk_state.value, "is_new_alert": d.is_new_alert,
                        "reason": d.trigger_reason.value})
    return records, tl
