"""Replay one workload timeline through the policy engine (shared by the ablation and sensitivity studies)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import OpticalHealthConfig, PolicyConfig, SystemConfig, load_sensor_config
from src.experiments.workloads import ScoreBank, Timeline, WorkloadConfig, build_timeline
from src.inference_service import InferenceEngine, InferenceResult
from src.runtime.alarm_baselines import make_engine
from src.sensor_simulator import MachineState, SensorSimulator


def simulate_workload(
    workload: WorkloadConfig, bank: ScoreBank, policy_cfg: PolicyConfig, unit: int, rho: float = 0.0
) -> Tuple[List[Dict[str, Any]], Timeline]:
    """Run ``workload`` for experimental unit ``unit`` and return per-step decision records.

    The timeline, visual-score sampling and sensor noise depend only on ``unit``, so every
    policy configuration sees exactly the same inputs for a given unit (paired design).

    ``rho = 0`` draws every visual score independently from the matching pool. ``rho > 0`` makes
    consecutive scores temporally correlated through a Gaussian copula: a latent AR(1) process
    z_t = rho z_{t-1} + sqrt(1 - rho^2) e_t is mapped through the standard normal CDF to a quantile
    of the current pool (good, defect or glare). The marginal distribution of each pool is
    unchanged; only the frame-to-frame dependence differs.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    tl = build_timeline(workload, seed=unit)
    rng = np.random.default_rng(10_000 + unit)
    engine = make_engine(policy_cfg)
    sorted_pools = {k: np.sort(v) for k, v in bank.pools.items()}
    z = float(rng.standard_normal()) if rho > 0 else 0.0

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
        if rho == 0.0:
            drawn = float(rng.choice(bank.pools[int(tl.vision_source[t])]))
        else:
            z = rho * z + np.sqrt(1.0 - rho * rho) * float(rng.standard_normal())
            pool = sorted_pools[int(tl.vision_source[t])]
            u = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            drawn = float(pool[min(len(pool) - 1, int(u * len(pool)))])
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
