"""Temporal decision policy and cross-modal risk engine.

The engine turns a stream of (visual score, sensor reading) pairs into risk
decisions. It implements exponential smoothing, k-of-N persistence, machine
state gating, cross-modal divergence triage, degraded-input fallbacks and an
incident latch that aggregates repeated triggers of one sustained condition
into a single operator alert. Policy modes switch individual stages off for
ablation experiments.
"""

from __future__ import annotations

import collections
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import PolicyConfig, PolicyMode, load_policy_config
from src.inference_service import InferenceResult
from src.sensor_simulator import MachineState, SensorReading


class RiskState(str, Enum):
    """Operational risk classification output by the policy engine."""
    NORMAL = "NORMAL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    HIGH_SEVERITY = "HIGH_SEVERITY"


_SEVERITY_RANK = {RiskState.NORMAL: 0, RiskState.REVIEW_REQUIRED: 1, RiskState.HIGH_SEVERITY: 2}


class TriggerReason(str, Enum):
    """Rule that produced the decision."""
    NOMINAL_OPERATION = "NOMINAL_OPERATION"
    SUSTAINED_VISION_ANOMALY = "SUSTAINED_VISION_ANOMALY"
    SUSTAINED_SENSOR_ANOMALY = "SUSTAINED_SENSOR_ANOMALY"
    MULTI_MODAL_CONFIRMED_FAULT = "MULTI_MODAL_CONFIRMED_FAULT"
    CROSS_MODAL_DISCREPANCY = "CROSS_MODAL_DISCREPANCY"
    OPTICAL_DEGRADATION_FALLBACK = "OPTICAL_DEGRADATION_FALLBACK"
    OPTICAL_TRANSIENT_HELD = "OPTICAL_TRANSIENT_HELD"
    SENSOR_DEGRADATION_FALLBACK = "SENSOR_DEGRADATION_FALLBACK"
    STATE_GATED_SUPPRESSION = "STATE_GATED_SUPPRESSION"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"  # retained for schema compatibility; no longer emitted
    CRITICAL_MACHINE_FAULT = "CRITICAL_MACHINE_FAULT"


class PolicyDecision(BaseModel):
    """Decision record emitted for every evaluated cycle."""
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique decision identifier.")
    source_id: str = Field(default="edge-gateway-01", description="Identifier of the originating edge node.")
    sequence_id: int = Field(default=0, ge=0, description="Per-source sequence number, incremented per decision.")
    created_monotonic_ns: int = Field(default_factory=time.monotonic_ns, description="Monotonic clock (ns).")
    schema_version: str = Field(default="1.1", description="Event contract version.")
    decision_id: str = Field(default="", description="Alias of event_id kept for older consumers.")
    timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp (YYYY-MM-DDTHH:MM:SS.fffZ).")
    camera_id: str = Field(..., description="Camera identifier.")
    machine_id: str = Field(..., description="Machine identifier.")
    machine_state: MachineState = Field(..., description="Machine state reported with the sensor reading.")
    risk_state: RiskState = Field(..., description="Risk classification of this cycle.")
    trigger_reason: TriggerReason = Field(..., description="Rule responsible for the classification.")
    raw_scores: Dict[str, float] = Field(..., description="Instantaneous scores.")
    smoothed_scores: Dict[str, float] = Field(..., description="Smoothed scores.")
    window_stats: Dict[str, Any] = Field(..., description="k-of-N window statistics.")
    cooldown_remaining: int = Field(..., ge=0, description="Quiet frames still required to close the open incident.")
    is_degraded: bool = Field(..., description="True if the optical stream or any sensor channel is degraded.")
    incident_id: Optional[str] = Field(default=None, description="Incident this decision belongs to, if any.")
    is_new_alert: bool = Field(default=False, description="True if this decision enqueues a new operator alert.")
    frame_id: Optional[str] = Field(default=None, description="Correlated frame identifier.")
    reading_id: Optional[str] = Field(default=None, description="Correlated sensor reading identifier.")
    evidence_uri: Optional[str] = Field(default=None, description="Path or URI of the evidence image.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Inference latency plus policy latency (ms).")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic values.")

    @model_validator(mode="after")
    def _sync_ids(self) -> "PolicyDecision":
        if not self.decision_id:
            self.decision_id = self.event_id
        elif self.decision_id != self.event_id:
            raise ValueError("decision_id must equal event_id when both are supplied")
        return self

    def to_mqtt_payload(self) -> Dict[str, Any]:
        """Serialize the decision following the event contract (docs/design/event-schema.md)."""
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        return payload


class TemporalPolicyEngine:
    """Temporal decision policy and cross-modal risk engine."""

    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        camera_id: str = "line1_overhead_cam01",
        machine_id: str = "press_unit_04",
        source_id: str = "edge-gateway-01",
    ) -> None:
        self.config = config or load_policy_config()
        self.camera_id = camera_id
        self.machine_id = machine_id
        self.source_id = source_id

        n_win = self.config.confirmation_window.window_size_n
        self.vision_ema: Optional[float] = None
        self.sensor_ema: Optional[float] = None
        self.vision_high_history: Deque[bool] = collections.deque(maxlen=n_win)
        self.vision_med_history: Deque[bool] = collections.deque(maxlen=n_win)
        self.sensor_history: Deque[bool] = collections.deque(maxlen=n_win)
        self.optical_invalid_history: Deque[bool] = collections.deque(maxlen=n_win)

        self._sequence_id = 0
        self._incident_id: Optional[str] = None
        self._incident_severity: RiskState = RiskState.NORMAL
        self._quiet_frames = 0

        self.total_evaluations = 0
        self.total_alerts = 0
        self.total_aggregated = 0
        self.total_state_suppressions = 0

        logger.info(
            f"Initialized TemporalPolicyEngine (mode={self.config.policy_mode.value}, "
            f"alpha_v={self.config.temporal_smoothing.alpha_vision}, "
            f"k_of_n={self.config.confirmation_window.required_k}/{n_win}, "
            f"cooldown={self.config.cooldown.cooldown_steps})"
        )

    # Backward-compatible counter names used by older scripts and tests.
    @property
    def total_escalations(self) -> int:
        return self.total_alerts

    @property
    def total_cooldown_suppressions(self) -> int:
        return self.total_aggregated

    @property
    def cooldown_counter(self) -> int:
        if self._incident_id is None:
            return 0
        return max(0, self.config.cooldown.cooldown_steps - self._quiet_frames)

    def reset(self) -> None:
        """Clear filters, windows, incident state and counters."""
        self.vision_ema = None
        self.sensor_ema = None
        for dq in (self.vision_high_history, self.vision_med_history, self.sensor_history, self.optical_invalid_history):
            dq.clear()
        self._sequence_id = 0
        self._incident_id = None
        self._incident_severity = RiskState.NORMAL
        self._quiet_frames = 0
        self.total_evaluations = 0
        self.total_alerts = 0
        self.total_aggregated = 0
        self.total_state_suppressions = 0

    def open_incident(self, severity: RiskState = RiskState.HIGH_SEVERITY) -> str:
        """Open an incident explicitly (used by tests and by operators acknowledging an alarm)."""
        self._incident_id = str(uuid.uuid4())
        self._incident_severity = severity
        self._quiet_frames = 0
        return self._incident_id

    def get_telemetry_stats(self) -> Dict[str, Any]:
        """Runtime counters."""
        return {
            "total_evaluations": self.total_evaluations,
            "total_alerts": self.total_alerts,
            "total_aggregated": self.total_aggregated,
            "total_state_suppressions": self.total_state_suppressions,
            "alert_rate": self.total_alerts / max(1, self.total_evaluations),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ema(prev: Optional[float], x: float, alpha: float) -> float:
        return float(x) if prev is None else float(alpha * x + (1.0 - alpha) * prev)

    def _mode_flags(self) -> Dict[str, bool]:
        mode = self.config.policy_mode
        cascade = mode in (
            PolicyMode.NO_COOLDOWN, PolicyMode.NO_FUSION, PolicyMode.NO_DIVERGENCE,
            PolicyMode.NO_STATE_GATING, PolicyMode.FULL_POLICY,
        )
        return {
            "smoothing": mode != PolicyMode.BASELINE,
            "persistence": mode not in (PolicyMode.BASELINE, PolicyMode.EMA_ONLY),
            "cascade": cascade,
            "fusion": cascade and mode != PolicyMode.NO_FUSION,
            "divergence": cascade and mode not in (PolicyMode.NO_FUSION, PolicyMode.NO_DIVERGENCE),
            "gating": cascade and mode != PolicyMode.NO_STATE_GATING,
            "incidents": cascade and mode != PolicyMode.NO_COOLDOWN,
        }

    def _classify(
        self,
        flags: Dict[str, bool],
        machine_state: MachineState,
        optical_invalid: bool,
        sensor_available: bool,
        raw_v: float,
    ) -> Tuple[RiskState, TriggerReason, Dict[str, Any]]:
        """Candidate classification before state gating and incident aggregation."""
        th = self.config.thresholds
        k = self.config.confirmation_window.required_k
        v = self.vision_ema if flags["smoothing"] else raw_v
        v_high_count = sum(self.vision_high_history)
        v_med_count = sum(self.vision_med_history)
        s_count = sum(self.sensor_history)

        if flags["persistence"]:
            vision_high = v_high_count >= k
            vision_med = v_med_count >= k
        else:
            vision_high = v is not None and v >= th.vision_high
            vision_med = v is not None and v >= th.vision_medium
        sensor_confirmed = flags["fusion"] and sensor_available and s_count >= k
        divergence = abs((v or 0.0) - (self.sensor_ema or 0.0))

        info = {
            "vision_confirmed_high": bool(vision_high),
            "vision_confirmed_medium": bool(vision_med),
            "sensor_confirmed": bool(sensor_confirmed),
            "cross_modal_divergence": float(divergence),
            "counts": {"vision_high": v_high_count, "vision_medium": v_med_count, "sensor_high": s_count,
                       "optical_invalid": sum(self.optical_invalid_history)},
        }

        if not flags["cascade"]:
            # Vision-only baselines: no health checks, no sensors, no machine state.
            if vision_high:
                return RiskState.HIGH_SEVERITY, TriggerReason.SUSTAINED_VISION_ANOMALY, info
            if vision_med:
                return RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_VISION_ANOMALY, info
            return RiskState.NORMAL, TriggerReason.NOMINAL_OPERATION, info

        if machine_state == MachineState.FAULT:
            return RiskState.HIGH_SEVERITY, TriggerReason.CRITICAL_MACHINE_FAULT, info

        if optical_invalid:
            # Vision is blind: only the physical channel can raise risk this cycle.
            if sensor_confirmed:
                return RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_SENSOR_ANOMALY, info
            if sum(self.optical_invalid_history) >= k:
                return RiskState.REVIEW_REQUIRED, TriggerReason.OPTICAL_DEGRADATION_FALLBACK, info
            return RiskState.NORMAL, TriggerReason.OPTICAL_TRANSIENT_HELD, info

        if vision_high and sensor_confirmed:
            return RiskState.HIGH_SEVERITY, TriggerReason.MULTI_MODAL_CONFIRMED_FAULT, info
        if vision_high:
            if flags["fusion"] and not sensor_available:
                return RiskState.REVIEW_REQUIRED, TriggerReason.SENSOR_DEGRADATION_FALLBACK, info
            if flags["divergence"] and divergence >= th.cross_modal_divergence:
                return RiskState.REVIEW_REQUIRED, TriggerReason.CROSS_MODAL_DISCREPANCY, info
            return RiskState.HIGH_SEVERITY, TriggerReason.SUSTAINED_VISION_ANOMALY, info
        if sensor_confirmed:
            return RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_SENSOR_ANOMALY, info
        if vision_med:
            if flags["divergence"] and divergence >= th.cross_modal_divergence:
                return RiskState.REVIEW_REQUIRED, TriggerReason.CROSS_MODAL_DISCREPANCY, info
            return RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_VISION_ANOMALY, info
        return RiskState.NORMAL, TriggerReason.NOMINAL_OPERATION, info

    # ------------------------------------------------------------------- public
    def evaluate(
        self,
        inference_result: InferenceResult,
        sensor_reading: SensorReading,
        evidence_uri: Optional[str] = None,
    ) -> PolicyDecision:
        """Evaluate one synchronized (frame, sensor reading) pair."""
        t0 = time.perf_counter()
        self.total_evaluations += 1
        flags = self._mode_flags()
        th = self.config.thresholds
        ts = self.config.temporal_smoothing

        raw_v = float(inference_result.vision_score)
        raw_s = float(sensor_reading.sensor_score)
        machine_state = sensor_reading.machine_state
        optical_invalid = flags["cascade"] and not inference_result.optical_health.is_valid
        sensor_available = not sensor_reading.is_degraded

        # 1) Update smoothed state. Invalid inputs are not allowed to move the filters.
        if not optical_invalid:
            self.vision_ema = self._ema(self.vision_ema, raw_v, ts.alpha_vision) if flags["smoothing"] else raw_v
            self.vision_high_history.append(self.vision_ema >= th.vision_high)
            self.vision_med_history.append(self.vision_ema >= th.vision_medium)
        if sensor_available:
            self.sensor_ema = self._ema(self.sensor_ema, raw_s, ts.alpha_sensor)
            self.sensor_history.append(self.sensor_ema >= th.sensor_anomaly)
        self.optical_invalid_history.append(bool(optical_invalid))

        # 2) Candidate classification.
        risk, reason, info = self._classify(flags, machine_state, optical_invalid, sensor_available, raw_v)

        # 3) Machine-state gating: lower one severity level while IDLE or in MAINTENANCE.
        gated = False
        if flags["gating"] and risk != RiskState.NORMAL:
            gating = self.config.machine_state_gating
            if (machine_state == MachineState.IDLE and gating.suppress_high_severity_on_idle) or (
                machine_state == MachineState.MAINTENANCE and gating.suppress_high_severity_on_maintenance
            ):
                gated = True
                self.total_state_suppressions += 1
                risk = RiskState.REVIEW_REQUIRED if risk == RiskState.HIGH_SEVERITY else RiskState.NORMAL
                reason = TriggerReason.STATE_GATED_SUPPRESSION

        # 4) Alert emission with or without incident aggregation.
        is_new_alert = False
        incident_id: Optional[str] = None
        if flags["incidents"]:
            if risk != RiskState.NORMAL:
                self._quiet_frames = 0
                if self._incident_id is None:
                    self.open_incident(risk)
                    is_new_alert = True
                elif _SEVERITY_RANK[risk] > _SEVERITY_RANK[self._incident_severity]:
                    self._incident_severity = risk
                    is_new_alert = True  # severity upgrade inside the same incident
                else:
                    self.total_aggregated += 1
                incident_id = self._incident_id
            elif self._incident_id is not None:
                self._quiet_frames += 1
                incident_id = self._incident_id
                if self._quiet_frames >= self.config.cooldown.cooldown_steps:
                    self._incident_id = None
                    self._incident_severity = RiskState.NORMAL
                    self._quiet_frames = 0
        elif risk != RiskState.NORMAL:
            is_new_alert = True
            incident_id = str(uuid.uuid4())

        if is_new_alert:
            self.total_alerts += 1

        seq = self._sequence_id
        self._sequence_id += 1
        policy_ms = (time.perf_counter() - t0) * 1000.0
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        n_win = self.config.confirmation_window.window_size_n

        return PolicyDecision(
            source_id=self.source_id,
            sequence_id=seq,
            timestamp_utc=now_utc,
            camera_id=self.camera_id,
            machine_id=self.machine_id,
            machine_state=machine_state,
            risk_state=risk,
            trigger_reason=reason,
            raw_scores={"vision_raw": raw_v, "sensor_raw": raw_s},
            smoothed_scores={
                "vision_ema": float(self.vision_ema if self.vision_ema is not None else raw_v),
                "sensor_ema": float(self.sensor_ema if self.sensor_ema is not None else raw_s),
            },
            window_stats={
                "window_size_n": n_win,
                "required_k": self.config.confirmation_window.required_k,
                "active_exceedances_count": info["counts"],
                "vision_confirmed_high": info["vision_confirmed_high"],
                "vision_confirmed_medium": info["vision_confirmed_medium"],
                "sensor_confirmed": info["sensor_confirmed"],
            },
            cooldown_remaining=self.cooldown_counter,
            is_degraded=bool(not inference_result.optical_health.is_valid or sensor_reading.is_degraded),
            incident_id=incident_id,
            is_new_alert=is_new_alert,
            frame_id=inference_result.frame_id,
            reading_id=sensor_reading.reading_id,
            evidence_uri=evidence_uri,
            latency_ms=float(inference_result.latency_ms + policy_ms),
            diagnostics={
                "policy_mode": self.config.policy_mode.value,
                "policy_latency_ms": policy_ms,
                "cross_modal_divergence": info["cross_modal_divergence"],
                "optical_degradation_reason": inference_result.optical_health.degradation_reason,
                "missing_sensor_channels": list(sensor_reading.missing_channels),
                "state_gated": gated,
            },
        )
