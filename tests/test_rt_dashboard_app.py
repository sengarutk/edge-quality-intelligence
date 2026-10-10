"""Integration tests for Streamlit dashboard application functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.audit_log import AuditLogDB
from src.config import AuditConfig, load_mqtt_config
from src.dashboard.app import (
    compute_dynamic_fmea,
    generate_defect_heatmap,
    generate_synthetic_industrial_frame,
    get_database,
    get_evidence_mgr,
    get_system_status,
    inject_chaos_fault_event,
    restore_system_nominal,
    seed_demo_simulation,
)
from src.evidence_manager import EvidenceManager
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


def test_get_system_status_helper() -> None:
    """Test get_system_status helper with various event configurations."""
    status, badge = get_system_status([])
    assert status == "OPERATIONAL"
    assert "healthy" in badge

    status, badge = get_system_status([{"risk_state": "HIGH_SEVERITY", "is_degraded": 0}])
    assert status == "CRITICAL LOCKOUT"
    assert "critical" in badge

    status, badge = get_system_status([{"risk_state": "REVIEW_REQUIRED", "is_degraded": 1}])
    assert status == "DEGRADED REVIEW"
    assert "warning" in badge

    status, badge = get_system_status([{"risk_state": "NORMAL", "is_degraded": 0}])
    assert status == "OPERATIONAL"
    assert "healthy" in badge


def test_factories_initialization(tmp_path: Path) -> None:
    """Test get_database and get_evidence_mgr factories."""
    db_file = tmp_path / "factory_audit.db"
    db = get_database(db_path=str(db_file))
    assert db is not None
    db.close()

    ev_mgr = get_evidence_mgr(storage_dir=str(tmp_path / "factory_ev"))
    assert ev_mgr is not None


def test_synthetic_generation_helpers() -> None:
    """Test industrial frame and defect heatmap generators."""
    frame = generate_synthetic_industrial_frame(width=224, height=224, seed=42)
    assert frame.shape == (224, 224, 3)
    assert frame.dtype == np.uint8

    heatmap = generate_defect_heatmap(width=224, height=224, center=(112, 112), radius=35)
    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32
    assert 0.99 <= float(heatmap.max()) <= 1.01
    assert 0.0 <= float(heatmap.min()) <= 0.05


def test_seed_demo_simulation(tmp_path: Path) -> None:
    """Test seed_demo_simulation helper populating database and evidence."""
    db_file = tmp_path / "seed_audit.db"
    ev_dir = tmp_path / "seed_ev"
    db = AuditLogDB(db_path=str(db_file))
    ev_mgr = EvidenceManager(storage_dir=str(ev_dir))

    count = seed_demo_simulation(n_steps=10, db=db, evidence_mgr=ev_mgr)
    assert count == 10

    events = db.query_recent_events(limit=50)
    assert len(events) == 10
    telemetry = db.query_recent_telemetry(limit=50)
    assert len(telemetry) == 10

    db.close()


def test_inject_chaos_and_restore(tmp_path: Path) -> None:
    """Test all chaos injection fault triggers and nominal restore in dashboard."""
    db_file = tmp_path / "chaos_audit.db"
    ev_dir = tmp_path / "chaos_ev"
    db = AuditLogDB(db_path=str(db_file))
    ev_mgr = EvidenceManager(storage_dir=str(ev_dir))

    inject_chaos_fault_event("OPTICAL_BLUR", db, ev_mgr)
    inject_chaos_fault_event("NETWORK_PARTITION", db, ev_mgr)
    inject_chaos_fault_event("SENSOR_DRIFT", db, ev_mgr)
    inject_chaos_fault_event("SENSOR_DROPOUT", db, ev_mgr)

    events = db.query_recent_events(limit=50)
    assert len(events) >= 3
    health_rows = db.query_recent_health(limit=50)
    assert len(health_rows) >= 4

    # Test Restore to Nominal
    restore_system_nominal(db)
    health_restored = db.query_recent_health(limit=50)
    assert len(health_restored) >= 9

    db.close()


def test_compute_dynamic_fmea(tmp_path: Path) -> None:
    """Test dynamic FMEA calculation under degraded and nominal conditions."""
    db_file = tmp_path / "fmea_audit.db"
    db = AuditLogDB(db_path=str(db_file))

    # Nominal FMEA
    rows_nominal = compute_dynamic_fmea(db, [], [])
    assert len(rows_nominal) == 5
    assert all("Subsystem" in r and "Health Status" in r for r in rows_nominal)

    # Degraded FMEA with blur and sensor dropout
    db.insert_system_health("camera_optics", "DEGRADED", "OPTICAL_BLUR_DETECTED")
    db.insert_system_health("mqtt_broker", "OFFLINE", "NETWORK_PARTITION")
    db.insert_system_health("current_sensor", "DEGRADED", "DROPOUT")

    rows_degraded = compute_dynamic_fmea(
        db,
        [{"is_degraded": 1, "trigger_reason": "OPTICAL_DEGRADATION_FALLBACK"}],
        [{"vibration_rms": 0.0, "current_amps": 0.0, "latency_ms": 30.0}],
    )
    assert len(rows_degraded) == 5
    assert "DEGRADED" in rows_degraded[0]["Health Status"]
    assert "DISCONNECTED" in rows_degraded[3]["Health Status"]

    db.close()


def test_triage_actions_and_empty_state(tmp_path: Path) -> None:
    """Test triage review actions and empty state rendering."""
    db_file = tmp_path / "triage_audit.db"
    db = AuditLogDB(db_path=str(db_file))

    # Insert actionable event
    d = PolicyDecision(
        timestamp_utc="2026-08-27T15:00:00.000Z",
        camera_id="line1_overhead_cam01",
        machine_id="press_unit_04",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.HIGH_SEVERITY,
        trigger_reason=TriggerReason.SUSTAINED_VISION_ANOMALY,
        raw_scores={"vision_raw": 0.95, "sensor_raw": 0.05},
        smoothed_scores={"vision_ema": 0.90, "sensor_ema": 0.05},
        window_stats={"persistence_count": 5},
        cooldown_remaining=15,
        is_degraded=False,
    )
    db.insert_risk_event(d)

    # Test Confirm action
    res_confirm = db.record_operator_review(d.decision_id, action="CONFIRMED", notes="Operator verified crack")
    assert res_confirm is True
    ev_confirmed = db.get_actionable_event_by_id(d.decision_id)
    assert ev_confirmed["review_status"] == "CONFIRMED"

    # Test Reject action
    res_reject = db.record_operator_review(d.decision_id, action="REJECTED", notes="False positive flicker")
    assert res_reject is True
    ev_rejected = db.get_actionable_event_by_id(d.decision_id)
    assert ev_rejected["review_status"] == "REJECTED"

    db.close()


def test_streamlit_app_rendering(tmp_path: Path) -> None:
    """Test Streamlit app rendering using AppTest with clean disconnect."""
    app_file = Path(__file__).parent.parent / "src" / "dashboard" / "app.py"

    at = AppTest.from_file(str(app_file), default_timeout=15)
    try:
        at.run(timeout=15)
        assert not at.exception
    finally:
        try:
            at.disconnect()
        except Exception:
            pass
