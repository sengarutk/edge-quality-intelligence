"""Unit tests for persistent SQLite audit database and operator triage."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.audit_log import AuditLogDB
from src.config import AuditConfig
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


@pytest.fixture
def audit_db(tmp_path: Path) -> AuditLogDB:
    """Fixture providing an isolated AuditLogDB instance in a temp directory."""
    db_file = tmp_path / "audit.db"
    cfg = AuditConfig(db_path=str(db_file))
    db = AuditLogDB(config=cfg)
    yield db
    db.close()


def test_audit_db_direct_path_init(tmp_path: Path) -> None:
    """Test initializing AuditLogDB with db_path directly."""
    db_file = tmp_path / "direct_audit.db"
    db = AuditLogDB(db_path=str(db_file))
    assert db.db_path == db_file
    db.close()


def test_insert_risk_event_and_query(audit_db: AuditLogDB) -> None:
    """Test inserting PolicyDecision and querying historical records."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    decision = PolicyDecision(
        timestamp_utc=now_utc,
        camera_id="line1_overhead_cam01",
        machine_id="press_unit_04",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.HIGH_SEVERITY,
        trigger_reason=TriggerReason.SUSTAINED_VISION_ANOMALY,
        raw_scores={"vision_raw": 0.88, "sensor_raw": 0.05},
        smoothed_scores={"vision_ema": 0.82, "sensor_ema": 0.04},
        window_stats={"window_size_n": 10, "consecutive_k": 4, "active_exceedances_count": {"vision_high": 4}},
        cooldown_remaining=15,
        is_degraded=False,
        is_new_alert=True,
        frame_id=str(uuid.uuid4()),
        reading_id=str(uuid.uuid4()),
        evidence_uri="/var/data/frames/frame_01.jpg",
        latency_ms=12.4,
        diagnostics={"test_diag": True},
    )

    event_id = audit_db.insert_risk_event(decision)
    assert event_id == decision.decision_id

    events = audit_db.query_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["risk_state"] == "HIGH_SEVERITY"
    assert events[0]["review_status"] == "PENDING"
    assert events[0]["evidence_uri"] == "/var/data/frames/frame_01.jpg"


def test_telemetry_and_health_insertion(audit_db: AuditLogDB) -> None:
    """Test inserting continuous telemetry stream and health entries."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    reading = SensorReading(
        timestamp_utc=now_utc,
        machine_id="press_unit_04",
        machine_state=MachineState.RUNNING,
        vibration_rms=0.46,
        temperature_c=62.5,
        current_amps=12.9,
        missing_channels=[],
        is_degraded=False,
        sensor_score=0.04,
        sensor_breakdown={},
    )
    inf_res = InferenceResult(
        timestamp_utc=now_utc,
        camera_id="line1_overhead_cam01",
        model_metadata={},
        vision_score=0.08,
        is_blurred=False,
        is_occluded=False,
        optical_health=OpticalHealthStatus(is_valid=True, laplacian_var=150.0, mean_brightness=128.0),
        heatmap=None,
        latency_ms=8.5,
    )

    t_id = audit_db.insert_telemetry(reading, inf_res)
    assert t_id > 0

    h_id = audit_db.insert_system_health("camera_stream", "HEALTHY", "Frame rate 30fps")
    assert h_id > 0


def test_operator_review_workflow_and_metrics(audit_db: AuditLogDB) -> None:
    """Test operator triage workflow (CONFIRMED vs REJECTED) and metric aggregates."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    d1 = PolicyDecision(
        timestamp_utc=now_utc,
        camera_id="line1",
        machine_id="press1",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.HIGH_SEVERITY,
        trigger_reason=TriggerReason.SUSTAINED_VISION_ANOMALY,
        raw_scores={"vision_raw": 0.9, "sensor_raw": 0.1},
        smoothed_scores={"vision_ema": 0.85, "sensor_ema": 0.05},
        window_stats={},
        cooldown_remaining=0,
        is_degraded=False,
        is_new_alert=True,
    )
    d2 = PolicyDecision(
        timestamp_utc=now_utc,
        camera_id="line1",
        machine_id="press1",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.HIGH_SEVERITY,
        trigger_reason=TriggerReason.SUSTAINED_VISION_ANOMALY,
        raw_scores={"vision_raw": 0.9, "sensor_raw": 0.1},
        smoothed_scores={"vision_ema": 0.85, "sensor_ema": 0.05},
        window_stats={},
        cooldown_remaining=0,
        is_degraded=False,
        is_new_alert=True,
    )
    d3 = PolicyDecision(
        timestamp_utc=now_utc,
        camera_id="line1",
        machine_id="press1",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.REVIEW_REQUIRED,
        trigger_reason=TriggerReason.CROSS_MODAL_DISCREPANCY,
        raw_scores={"vision_raw": 0.6, "sensor_raw": 0.1},
        smoothed_scores={"vision_ema": 0.55, "sensor_ema": 0.05},
        window_stats={},
        cooldown_remaining=0,
        is_degraded=False,
        is_new_alert=True,
    )

    audit_db.insert_risk_event(d1)
    audit_db.insert_risk_event(d2)
    audit_db.insert_risk_event(d3)

    metrics_initial = audit_db.get_operator_metrics()
    assert metrics_initial["total_actionable_events"] == 3
    assert metrics_initial["pending_reviews"] == 3
    assert metrics_initial["confirmed_defects"] == 0

    assert audit_db.record_operator_review(d1.decision_id, "CONFIRMED", "Confirmed weld defect") is True
    assert audit_db.record_operator_review(d2.decision_id, "REJECTED", "False positive lighting artifact") is True

    with pytest.raises(ValueError, match="Invalid review action"):
        audit_db.record_operator_review(d3.decision_id, "INVALID_ACTION")

    metrics_after = audit_db.get_operator_metrics()
    assert metrics_after["pending_reviews"] == 1
    assert metrics_after["confirmed_defects"] == 1
    assert metrics_after["rejected_false_positives"] == 1
    assert metrics_after["confirmation_rate"] == 0.50

    high_events = audit_db.query_recent_events(limit=10, risk_filter="HIGH_SEVERITY")
    assert len(high_events) == 2