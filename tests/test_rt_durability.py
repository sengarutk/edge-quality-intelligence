"""Regression tests for the spool and audit-log durability fixes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.audit_log import AuditLogDB
from src.config import SpoolerConfig
from src.policy import TemporalPolicyEngine
from src.spooler import DiskSpooler
from tests.helpers import inf, reading


def spool(tmp_path: Path, cap: int = 100) -> DiskSpooler:
    return DiskSpooler(config=SpoolerConfig(db_path=str(tmp_path / "s.db"), max_spool_records=cap))


def test_spool_deduplicates_on_event_id(tmp_path):
    sp = spool(tmp_path)
    assert sp.enqueue("t", json.dumps({"event_id": "a"})) is True
    assert sp.enqueue("t", json.dumps({"event_id": "a"})) is False
    assert sp.get_queue_depth() == 1 == sp.count_on_disk() and sp.duplicate_count == 1


def test_spool_keeps_non_json_payloads(tmp_path):
    sp = spool(tmp_path)
    assert sp.enqueue("t", "raw-1") and sp.enqueue("t", "raw-1")
    assert sp.get_queue_depth() == 2


def test_spool_eviction_is_fifo_and_counted(tmp_path):
    sp = spool(tmp_path, cap=5)
    for i in range(8):
        sp.enqueue("t", json.dumps({"event_id": str(i)}))
    assert sp.get_queue_depth() == 5 and sp.evicted_count == 3
    assert json.loads(sp.peek_batch(1)[0][2])["event_id"] == "3"


def test_spool_survives_reopen(tmp_path):
    sp = spool(tmp_path)
    sp.enqueue("t", json.dumps({"event_id": "a"}))
    sp.close()
    assert spool(tmp_path).get_queue_depth() == 1


def test_audit_redelivery_does_not_erase_a_review(tmp_path):
    db = AuditLogDB(db_path=str(tmp_path / "a.db"))
    e = TemporalPolicyEngine()
    alert = next(d for d in (e.evaluate(inf(0.95), reading(0.95)) for _ in range(40)) if d.is_new_alert)
    db.insert_risk_event(alert)
    assert db.record_operator_review(alert.event_id, action="CONFIRMED", notes="ok")
    db.insert_risk_event(alert.to_mqtt_payload())  # QoS 1 redelivery
    row = db.get_actionable_event_by_id(alert.event_id)
    assert row["review_status"] == "CONFIRMED" and row["operator_notes"] == "ok"
    assert [r["event_id"] for r in db.query_pending_alerts()] == []


def test_audit_only_new_alerts_enter_the_queue(tmp_path):
    db = AuditLogDB(db_path=str(tmp_path / "a.db"))
    e = TemporalPolicyEngine()
    for _ in range(60):
        db.insert_risk_event(e.evaluate(inf(0.95), reading(0.95)))
    m = db.get_operator_metrics()
    assert m["total_actionable_events"] == e.total_alerts == len(db.query_pending_alerts())
    assert m["confirmation_rate"] is None


def test_audit_review_requires_valid_action(tmp_path):
    import pytest

    db = AuditLogDB(db_path=str(tmp_path / "a.db"))
    with pytest.raises(ValueError):
        db.record_operator_review("x")
    with pytest.raises(ValueError):
        db.record_operator_review("x", action="MAYBE")
    assert db.record_operator_review("x", review_status="rejected") is False  # unknown id, valid action


def test_audit_migrates_old_schema(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE risk_events (event_id TEXT PRIMARY KEY, timestamp_utc TEXT NOT NULL, camera_id TEXT NOT NULL,"
                " machine_id TEXT NOT NULL, machine_state TEXT NOT NULL, risk_state TEXT NOT NULL,"
                " trigger_reason TEXT NOT NULL, raw_payload TEXT, review_status TEXT DEFAULT 'PENDING',"
                " operator_notes TEXT, reviewed_at TEXT)")
    con.commit()
    con.close()
    db = AuditLogDB(db_path=str(path))
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(risk_events)")}
    assert {"source_id", "sequence_id", "incident_id", "is_new_alert"} <= cols
