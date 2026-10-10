"""Contract tests for the temporal policy engine.

Each test states one rule of the policy (see docs/design/policy-design.md) and checks it
with a short, hand-traceable input sequence under the default configuration
(alpha_v = 0.35, alpha_s = 0.25, k = 4, N = 10, T_cool = 15, tau_med = 0.5,
tau_high = 0.8, tau_phys = 0.7, tau_div = 0.45).
"""

from __future__ import annotations

from typing import List

import pytest

from src.config import ConfirmationWindowConfig, PolicyConfig, PolicyMode, ThresholdsConfig
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState
from tests.helpers import inf, reading


def engine(mode: PolicyMode = PolicyMode.FULL_POLICY, **kw) -> TemporalPolicyEngine:
    return TemporalPolicyEngine(config=PolicyConfig(policy_mode=mode, **kw))


def run(e: TemporalPolicyEngine, n: int, v: float, s: float = 0.05, **kw) -> List[PolicyDecision]:
    state = kw.pop("state", MachineState.RUNNING)
    valid = kw.pop("valid", True)
    missing = kw.pop("missing", None)
    return [e.evaluate(inf(v, valid=valid), reading(s, state=state, missing=missing)) for _ in range(n)]


def alerts(ds: List[PolicyDecision]) -> List[PolicyDecision]:
    return [d for d in ds if d.is_new_alert]


# --------------------------------------------------------------------- basics
def test_nominal_stream_raises_nothing():
    ds = run(engine(), 300, 0.05)
    assert all(d.risk_state == RiskState.NORMAL for d in ds)
    assert not alerts(ds)


def test_sequence_ids_are_monotonic_and_ids_unique():
    ds = run(engine(), 20, 0.05)
    assert [d.sequence_id for d in ds] == list(range(20))
    assert len({d.event_id for d in ds}) == 20
    assert all(d.decision_id == d.event_id for d in ds)


def test_decision_id_mismatch_is_rejected():
    with pytest.raises(ValueError):
        PolicyDecision(event_id="a", decision_id="b", timestamp_utc="t", camera_id="c", machine_id="m",
                       machine_state=MachineState.RUNNING, risk_state=RiskState.NORMAL,
                       trigger_reason=TriggerReason.NOMINAL_OPERATION, raw_scores={}, smoothed_scores={},
                       window_stats={}, cooldown_remaining=0, is_degraded=False)


def test_mqtt_payload_round_trips():
    d = run(engine(), 1, 0.05)[0]
    payload = d.to_mqtt_payload()
    assert payload["event_id"] == d.event_id and "decision_id" not in payload
    assert payload["risk_state"] == "NORMAL" and payload["is_new_alert"] is False


# ------------------------------------------------------------ persistence
def test_short_spike_is_suppressed_by_k_of_n():
    e = engine()
    run(e, 20, 0.05)
    ds = run(e, 1, 0.95) + run(e, 30, 0.05)
    assert not alerts(ds)


def test_baseline_alerts_on_every_single_frame_exceedance():
    e = engine(PolicyMode.BASELINE)
    ds = run(e, 3, 0.95)
    assert len(alerts(ds)) == 3 and all(d.risk_state == RiskState.HIGH_SEVERITY for d in ds)


def test_k_of_n_does_not_require_consecutive_samples():
    e = engine(PolicyMode.EMA_KOFN, temporal_smoothing={"alpha_vision": 1.0, "alpha_sensor": 1.0})
    ds = []
    for v in (0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9):
        ds += run(e, 1, v)
    assert ds[-1].risk_state == RiskState.HIGH_SEVERITY  # 4 exceedances within the last 10 samples


def test_k_must_not_exceed_n():
    with pytest.raises(ValueError):
        ConfirmationWindowConfig(window_size_n=3, required_k=4)


def test_threshold_ordering_is_validated():
    with pytest.raises(ValueError):
        ThresholdsConfig(vision_medium=0.9, vision_high=0.8)


# ---------------------------------------------------------- fusion rules
def test_corroborated_defect_escalates_high_once():
    e = engine()
    ds = run(e, 40, 0.95, s=0.95)
    high = [d for d in ds if d.risk_state == RiskState.HIGH_SEVERITY]
    assert high and high[0].trigger_reason == TriggerReason.MULTI_MODAL_CONFIRMED_FAULT
    assert len(alerts(ds)) == 1, "a sustained condition must produce exactly one alert"
    assert len({d.incident_id for d in ds if d.incident_id}) == 1


def test_persistent_visual_anomaly_without_sensor_evidence_goes_to_review():
    """Cosmetic glint: vision high for 2 s, sensors nominal -> review, never HIGH."""
    e = engine()
    run(e, 20, 0.05)
    ds = run(e, 60, 0.9)
    assert all(d.risk_state != RiskState.HIGH_SEVERITY for d in ds)
    assert alerts(ds) and alerts(ds)[0].trigger_reason == TriggerReason.CROSS_MODAL_DISCREPANCY


def test_no_divergence_mode_escalates_vision_only_anomaly():
    e = engine(PolicyMode.NO_DIVERGENCE)
    run(e, 20, 0.05)
    ds = run(e, 60, 0.9)
    assert any(d.risk_state == RiskState.HIGH_SEVERITY and d.trigger_reason == TriggerReason.SUSTAINED_VISION_ANOMALY
               for d in ds)


def test_sensor_only_anomaly_routes_to_review():
    ds = run(engine(), 40, 0.05, s=0.95)
    assert alerts(ds)[0].trigger_reason == TriggerReason.SUSTAINED_SENSOR_ANOMALY
    assert all(d.risk_state != RiskState.HIGH_SEVERITY for d in ds)


def test_no_fusion_mode_ignores_sensors():
    ds = run(engine(PolicyMode.NO_FUSION), 40, 0.05, s=0.95)
    assert not alerts(ds)


def test_degraded_sensor_prevents_corroboration():
    e = engine()
    ds = run(e, 40, 0.95, s=0.95, missing=["current"])
    assert all(d.risk_state != RiskState.HIGH_SEVERITY for d in ds)
    assert alerts(ds)[0].trigger_reason == TriggerReason.SENSOR_DEGRADATION_FALLBACK


def test_degraded_sensor_readings_do_not_move_the_sensor_filter():
    e = engine()
    run(e, 10, 0.05, s=0.05)
    before = e.sensor_ema
    run(e, 10, 0.05, s=0.99, missing=["vibration"])
    assert e.sensor_ema == before


# --------------------------------------------------------- optical health
def test_transient_blur_is_held_without_alert():
    e = engine()
    run(e, 20, 0.05)
    ds = run(e, 2, 0.0, valid=False) + run(e, 20, 0.05)
    assert not alerts(ds)
    assert ds[0].trigger_reason == TriggerReason.OPTICAL_TRANSIENT_HELD and ds[0].is_degraded


def test_sustained_blur_raises_one_review_alert():
    e = engine()
    ds = run(e, 60, 0.0, valid=False)
    assert [a.trigger_reason for a in alerts(ds)] == [TriggerReason.OPTICAL_DEGRADATION_FALLBACK]
    assert all(a.risk_state == RiskState.REVIEW_REQUIRED for a in alerts(ds))


def test_invalid_frames_do_not_move_the_vision_filter():
    e = engine()
    run(e, 10, 0.05)
    before = e.vision_ema
    run(e, 5, 0.99, valid=False)
    assert e.vision_ema == before


def test_blind_camera_with_mechanical_fault_state_escalates_once():
    e = engine()
    ds = run(e, 30, 0.0, s=0.95, valid=False, state=MachineState.FAULT)
    assert alerts(ds)[0].trigger_reason == TriggerReason.CRITICAL_MACHINE_FAULT
    assert len(alerts(ds)) == 1


# ----------------------------------------------------------- state gating
@pytest.mark.parametrize("state", [MachineState.IDLE, MachineState.MAINTENANCE])
def test_gating_lowers_severity_by_one_level(state):
    ds = run(engine(), 40, 0.95, s=0.95, state=state)
    assert all(d.risk_state != RiskState.HIGH_SEVERITY for d in ds)
    assert any(d.risk_state == RiskState.REVIEW_REQUIRED and d.trigger_reason == TriggerReason.STATE_GATED_SUPPRESSION
               for d in ds)


def test_gating_suppresses_review_level_evidence_completely():
    ds = run(engine(), 40, 0.9, s=0.05, state=MachineState.MAINTENANCE)
    assert not alerts(ds)


def test_no_state_gating_mode_keeps_severity():
    ds = run(engine(PolicyMode.NO_STATE_GATING), 40, 0.95, s=0.95, state=MachineState.MAINTENANCE)
    assert any(d.risk_state == RiskState.HIGH_SEVERITY for d in ds)


def test_fault_state_is_not_gated_and_is_aggregated():
    ds = run(engine(), 50, 0.05, state=MachineState.FAULT)
    assert all(d.risk_state == RiskState.HIGH_SEVERITY for d in ds)
    assert len(alerts(ds)) == 1


# ------------------------------------------------------- incident latching
def test_incident_closes_after_quiet_period_and_reopens():
    e = engine()
    first = run(e, 40, 0.95, s=0.95)
    quiet = run(e, 60, 0.05, s=0.05)
    second = run(e, 40, 0.95, s=0.95)
    assert len(alerts(first)) == 1 and not alerts(quiet)
    # The second episode opens a new incident; it may start at REVIEW (vision confirms first)
    # and upgrade to HIGH once the slower sensor filter corroborates, i.e. at most two alerts.
    assert 1 <= len(alerts(second)) <= 2
    assert {a.incident_id for a in alerts(second)} == {alerts(second)[0].incident_id}
    assert alerts(first)[0].incident_id != alerts(second)[0].incident_id
    assert alerts(second)[-1].risk_state == RiskState.HIGH_SEVERITY


def test_severity_upgrade_inside_incident_is_a_new_alert():
    e = engine()
    run(e, 20, 0.05)
    review = run(e, 30, 0.95, s=0.05)       # vision only -> review
    upgrade = run(e, 40, 0.95, s=0.95)      # sensors join -> high
    assert len(alerts(review)) == 1 and len(alerts(upgrade)) == 1
    assert alerts(upgrade)[0].risk_state == RiskState.HIGH_SEVERITY
    assert alerts(upgrade)[0].incident_id == alerts(review)[0].incident_id


def test_no_cooldown_mode_alerts_every_non_normal_frame():
    ds = run(engine(PolicyMode.NO_COOLDOWN), 40, 0.95, s=0.95)
    non_normal = [d for d in ds if d.risk_state != RiskState.NORMAL]
    assert len(alerts(ds)) == len(non_normal) > 1


def test_counters_and_reset():
    e = engine()
    run(e, 40, 0.95, s=0.95)
    stats = e.get_telemetry_stats()
    assert stats["total_alerts"] == 1 and stats["total_aggregated"] > 0
    e.reset()
    assert e.get_telemetry_stats()["total_evaluations"] == 0 and e.vision_ema is None
    assert run(e, 1, 0.05)[0].sequence_id == 0


def test_glare_tail_does_not_escalate_to_high():
    """Regression: after a burst ends, the k-of-N window still holds high samples while the smoothed
    score decays below tau_div away from the sensor score; that must not produce a HIGH alert."""
    e = engine()
    run(e, 30, 0.05)
    ds = run(e, 12, 1.0) + run(e, 30, 0.05)
    assert all(d.risk_state != RiskState.HIGH_SEVERITY for d in ds)


def test_partially_corroborated_high_vision_still_escalates():
    e = engine(temporal_smoothing={"alpha_vision": 1.0, "alpha_sensor": 1.0})
    ds = run(e, 12, 0.9, s=0.6)  # sensors elevated but not confirmed, |v - s| = 0.3 < tau_div
    assert ds[-1].risk_state == RiskState.HIGH_SEVERITY


def test_external_baselines_are_not_policy_variants():
    with pytest.raises(ValueError):
        engine(PolicyMode.DELAY_TIMER)
