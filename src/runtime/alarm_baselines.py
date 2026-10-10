"""External alarm-management baselines evaluated on the same inputs as the temporal policy.

These are textbook alarm designs (ISA-18.2 / EEMUA 191 style), not variants of our policy. They
reuse the thresholds of configs/policy_config.yaml so that no parameter is tuned for them:

DELAY_TIMER      on-delay / off-delay timers on the raw visual score. A level (REVIEW at tau_med,
                 HIGH at tau_high) becomes active after k consecutive samples at or above its
                 threshold and inactive after T_cool consecutive samples below it.
EMA_HYSTERESIS   the EMA-smoothed visual score with a deadband: a level becomes active when the
                 smoothed score reaches its threshold and inactive when it falls below the
                 threshold minus ``DEADBAND``.
DECISION_FUSION  decision-level fusion of two delay-timer alarms, one on the raw visual score
                 (tau_med) and one on the raw sensor score (tau_phys): HIGH if both are active,
                 REVIEW if one is.

All three are alarms with state: an alert is emitted when the alarm severity rises (activation or
escalation), which is the same alert definition used for the incident latch. None of them uses
the optical check, sensor-dropout flags or machine state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import PolicyConfig, PolicyMode
from src.inference_service import InferenceResult
from src.policy import RiskState, TriggerReason
from src.sensor_simulator import SensorReading

EXTERNAL_BASELINES = (PolicyMode.DELAY_TIMER, PolicyMode.EMA_HYSTERESIS, PolicyMode.DECISION_FUSION)
DEADBAND = 0.1  # fixed a priori: 10% of the normalized score range

_RANK = {RiskState.NORMAL: 0, RiskState.REVIEW_REQUIRED: 1, RiskState.HIGH_SEVERITY: 2}


@dataclass
class BaselineDecision:
    risk_state: RiskState
    trigger_reason: TriggerReason
    is_new_alert: bool


class _DelayTimer:
    """Boolean alarm with an on-delay of ``n_on`` and an off-delay of ``n_off`` samples."""

    def __init__(self, threshold: float, n_on: int, n_off: int) -> None:
        self.threshold, self.n_on, self.n_off = threshold, n_on, n_off
        self.active = False
        self._above = 0
        self._below = 0

    def update(self, x: float) -> bool:
        if x >= self.threshold:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0
        if not self.active and self._above >= self.n_on:
            self.active = True
        elif self.active and self._below >= self.n_off:
            self.active = False
        return self.active


class _Hysteresis:
    """Boolean alarm that activates at ``on`` and deactivates below ``off`` (< on)."""

    def __init__(self, on: float, off: float) -> None:
        self.on, self.off = on, off
        self.active = False

    def update(self, x: float) -> bool:
        if not self.active and x >= self.on:
            self.active = True
        elif self.active and x < self.off:
            self.active = False
        return self.active


class AlarmBaselineEngine:
    """Evaluates one of the EXTERNAL_BASELINES with the same call signature as TemporalPolicyEngine."""

    def __init__(self, config: PolicyConfig) -> None:
        mode = config.policy_mode
        if mode not in EXTERNAL_BASELINES:
            raise ValueError(f"{mode} is not an external baseline")
        self.mode = mode
        th = config.thresholds
        k = config.confirmation_window.required_k
        t_off = max(1, config.cooldown.cooldown_steps)
        self.alpha = config.temporal_smoothing.alpha_vision
        self.ema: Optional[float] = None
        if mode == PolicyMode.DELAY_TIMER:
            self.review = _DelayTimer(th.vision_medium, k, t_off)
            self.high = _DelayTimer(th.vision_high, k, t_off)
        elif mode == PolicyMode.EMA_HYSTERESIS:
            self.review = _Hysteresis(th.vision_medium, th.vision_medium - DEADBAND)
            self.high = _Hysteresis(th.vision_high, th.vision_high - DEADBAND)
        else:
            self.vision = _DelayTimer(th.vision_medium, k, t_off)
            self.sensor = _DelayTimer(th.sensor_anomaly, k, t_off)
        self._severity = RiskState.NORMAL

    def evaluate(self, inference_result: InferenceResult, sensor_reading: SensorReading,
                 evidence_uri: Optional[str] = None) -> BaselineDecision:
        v = float(inference_result.vision_score)
        if self.mode == PolicyMode.DECISION_FUSION:
            va = self.vision.update(v)
            sa = self.sensor.update(float(sensor_reading.sensor_score))
            if va and sa:
                risk, reason = RiskState.HIGH_SEVERITY, TriggerReason.MULTI_MODAL_CONFIRMED_FAULT
            elif va:
                risk, reason = RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_VISION_ANOMALY
            elif sa:
                risk, reason = RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_SENSOR_ANOMALY
            else:
                risk, reason = RiskState.NORMAL, TriggerReason.NOMINAL_OPERATION
        else:
            if self.mode == PolicyMode.EMA_HYSTERESIS:
                self.ema = v if self.ema is None else self.alpha * v + (1.0 - self.alpha) * self.ema
                x = self.ema
            else:
                x = v
            hi, rev = self.high.update(x), self.review.update(x)
            if hi:
                risk, reason = RiskState.HIGH_SEVERITY, TriggerReason.SUSTAINED_VISION_ANOMALY
            elif rev:
                risk, reason = RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_VISION_ANOMALY
            else:
                risk, reason = RiskState.NORMAL, TriggerReason.NOMINAL_OPERATION
        new_alert = _RANK[risk] > _RANK[self._severity]
        self._severity = risk
        return BaselineDecision(risk_state=risk, trigger_reason=reason, is_new_alert=new_alert)


def make_engine(config: PolicyConfig):
    """TemporalPolicyEngine for our policy variants, AlarmBaselineEngine for external baselines."""
    if config.policy_mode in EXTERNAL_BASELINES:
        return AlarmBaselineEngine(config)
    from src.policy import TemporalPolicyEngine

    return TemporalPolicyEngine(config=config)
