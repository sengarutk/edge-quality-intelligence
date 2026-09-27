"""Small constructors shared by the runtime tests."""

from __future__ import annotations

from typing import List, Optional

from src.inference_service import InferenceResult, OpticalHealthStatus
from src.sensor_simulator import MachineState, SensorReading

TS = "2026-01-01T00:00:00.000Z"


def inf(score: float = 0.05, valid: bool = True, reason: Optional[str] = None, latency_ms: float = 1.0) -> InferenceResult:
    if not valid and reason is None:
        reason = "OPTICAL_BLURRED"
    return InferenceResult(
        timestamp_utc=TS, camera_id="cam", vision_score=score,
        is_blurred=reason == "OPTICAL_BLURRED",
        is_occluded=reason in ("OPTICAL_OCCLUDED_DARK", "OPTICAL_OCCLUDED_BRIGHT"),
        optical_health=OpticalHealthStatus(is_valid=valid, laplacian_var=300.0 if valid else 10.0,
                                           mean_brightness=120.0, degradation_reason=reason),
        latency_ms=latency_ms,
    )


def reading(score: float = 0.05, state: MachineState = MachineState.RUNNING,
            missing: Optional[List[str]] = None) -> SensorReading:
    missing = missing or []
    return SensorReading(timestamp_utc=TS, machine_id="m", machine_state=state, vibration_rms=0.45,
                         temperature_c=62.0, current_amps=12.8, missing_channels=missing,
                         is_degraded=bool(missing), sensor_score=score)
