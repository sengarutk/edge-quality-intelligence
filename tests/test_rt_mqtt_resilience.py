"""Tests for the store-and-forward publisher, the subscriber and a real-broker round trip."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest

from src.audit_log import AuditLogDB
from src.config import AuditConfig, MQTTBrokerConfig, MQTTConfig, SpoolerConfig
from src.mqtt_publisher import ResilientMQTTPublisher
from src.mqtt_subscriber import MQTTEventSubscriber
from src.spooler import DiskSpooler

RISK = "inspection/line1/risk"


@pytest.fixture
def spool(tmp_path: Path) -> DiskSpooler:
    sp = DiskSpooler(config=SpoolerConfig(db_path=str(tmp_path / "spool.db"), max_spool_records=100))
    yield sp
    sp.close()


@pytest.fixture
def audit(tmp_path: Path) -> AuditLogDB:
    db = AuditLogDB(config=AuditConfig(db_path=str(tmp_path / "audit.db")))
    yield db
    db.close()


def info(rc: int = 0, mid: int = 1) -> MagicMock:
    m = MagicMock()
    m.rc, m.mid = rc, mid
    return m


# ------------------------------------------------------------- publisher
def test_qos1_events_are_spooled_before_any_network_io(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    pub._client.publish = MagicMock()
    assert pub.publish_event(RISK, {"event_id": "e1"}) is True
    assert spool.get_queue_depth() == 1 and not pub._client.publish.called


def test_duplicate_event_id_is_not_spooled_twice(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    assert pub.publish_event(RISK, {"event_id": "e1"}) is True
    assert pub.publish_event(RISK, {"event_id": "e1"}) is False
    assert spool.get_queue_depth() == 1 and pub.stats["deduplicated"] == 1


def test_record_is_deleted_only_after_acknowledgement(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    pub._is_connected = True
    pub._client.publish = MagicMock(side_effect=[info(0, 11), info(0, 12)])
    pub.publish_event(RISK, {"event_id": "a"})
    pub.publish_event(RISK, {"event_id": "b"})
    assert pub._drain_once() == 2
    assert spool.get_queue_depth() == 2, "handing a message to paho is not a delivery"
    pub._client.on_publish(pub._client, None, 11, None, None)
    assert spool.get_queue_depth() == 1
    remaining = json.loads(spool.peek_batch(1)[0][2])
    assert remaining["event_id"] == "b"


def test_acknowledgement_arriving_before_mapping_is_not_lost(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    pub._is_connected = True

    def publish_and_ack(topic, payload, qos):
        pub._client.on_publish(pub._client, None, 7, None, None)  # ack races ahead of the mapping
        return info(0, 7)

    pub._client.publish = MagicMock(side_effect=publish_and_ack)
    pub.publish_event(RISK, {"event_id": "a"})
    pub._drain_once()
    assert spool.get_queue_depth() == 0


def test_failed_publish_keeps_record_in_order(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    pub._is_connected = True
    pub._client.publish = MagicMock(return_value=info(mqtt.MQTT_ERR_NO_CONN, 0))
    pub.publish_event(RISK, {"event_id": "a"})
    assert pub._drain_once() == 0 and spool.get_queue_depth() == 1


def test_nothing_is_drained_while_offline(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    pub._client.publish = MagicMock()
    pub.publish_event(RISK, {"event_id": "a"})
    assert pub._drain_once() == 0 and not pub._client.publish.called


def test_qos0_is_best_effort(spool):
    cfg = MQTTConfig()
    pub = ResilientMQTTPublisher(config=cfg, spooler=spool)
    assert pub.publish_event(cfg.topics.telemetry, {"v": 1}) is False
    assert pub.stats["qos0_dropped"] == 1 and spool.get_queue_depth() == 0


def test_topic_specific_qos(spool):
    cfg = MQTTConfig()
    pub = ResilientMQTTPublisher(config=cfg, spooler=spool)
    assert pub._qos_for_topic(cfg.topics.heartbeat) == cfg.qos.heartbeat
    assert pub._qos_for_topic(cfg.topics.health) == cfg.qos.health


def test_pause_blocks_draining_until_resume(spool):
    pub = ResilientMQTTPublisher(config=MQTTConfig(), spooler=spool)
    pub._client.disconnect = MagicMock()
    pub._client.loop_stop = MagicMock()
    pub._client.connect_async = MagicMock()
    pub._client.loop_start = MagicMock()
    pub._is_connected = True
    pub.pause_network()
    assert not pub.is_connected
    pub.resume_network()
    assert pub.is_connected and pub._client.connect_async.called


@pytest.mark.parametrize("clean_session", [True, False])
def test_ack_after_reconnect_still_deletes_row(spool, clean_session):
    # paho re-sends unacknowledged messages after a reconnect under their original mid,
    # with or without clean_session, so the late PUBACK must still find its row.
    cfg = MQTTConfig(broker=MQTTBrokerConfig(clean_session=clean_session))
    pub = ResilientMQTTPublisher(config=cfg, spooler=spool)
    pub._is_connected = True
    pub._client.publish = MagicMock(return_value=info(0, 5))
    pub.publish_event(RISK, {"event_id": "a"})
    assert pub._drain_once() == 1
    pub._client.on_disconnect(pub._client, None, None, 1, None)
    pub._client.on_connect(pub._client, None, None, 0, None)
    assert pub._drain_once() == 0, "a row already in paho's queue must not be sent twice"
    pub._client.on_publish(pub._client, None, 5, None, None)
    assert spool.get_queue_depth() == 0 and not pub._early_acks


def test_late_ack_does_not_delete_row_that_reuses_its_mid(spool):
    """Regression: a PUBACK for a forgotten mid must not ack a later message given the same mid."""
    pub = ResilientMQTTPublisher(config=MQTTConfig(broker=MQTTBrokerConfig(clean_session=True)), spooler=spool)
    pub._is_connected = True
    pub._client.publish = MagicMock(side_effect=[info(0, 9), info(0, 9)])
    pub.publish_event(RISK, {"event_id": "a"})
    pub._drain_once()
    pub._client.on_disconnect(pub._client, None, None, 1, None)
    pub._client.on_connect(pub._client, None, None, 0, None)
    pub._client.on_publish(pub._client, None, 9, None, None)  # ack for "a" after reconnect
    pub.publish_event(RISK, {"event_id": "b"})
    pub._drain_once()  # paho's mid counter wrapped around to 9 again
    assert spool.get_queue_depth() == 1, "row 'b' was deleted without a broker acknowledgement"


def test_backoff_config_is_validated():
    with pytest.raises(ValueError):
        MQTTBrokerConfig(reconnect_delay_min_s=10.0, reconnect_delay_max_s=2.0)


# ------------------------------------------------------------ subscriber
def _msg(topic: str, payload) -> MagicMock:
    m = MagicMock()
    m.topic = topic
    m.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return m


RISK_EVENT = {
    "event_id": "evt-1", "timestamp_utc": "2026-01-01T00:00:00.000Z", "camera_id": "c", "machine_id": "m",
    "machine_state": "RUNNING", "risk_state": "HIGH_SEVERITY", "trigger_reason": "MULTI_MODAL_CONFIRMED_FAULT",
    "raw_scores": {}, "smoothed_scores": {}, "cooldown_remaining": 15, "is_degraded": False, "is_new_alert": True,
}


def test_subscriber_ingests_and_deduplicates(audit):
    seen = []
    sub = MQTTEventSubscriber(audit_db=audit, on_event_callback=lambda t, d: seen.append(t))
    sub._client.on_message(sub._client, None, _msg(RISK, RISK_EVENT))
    audit.record_operator_review("evt-1", action="CONFIRMED")
    sub._client.on_message(sub._client, None, _msg(RISK, RISK_EVENT))  # QoS 1 redelivery
    sub._client.on_message(sub._client, None, _msg("inspection/line1/health", {"component": "cam", "status": "OK"}))
    sub._client.on_message(sub._client, None, _msg(RISK, b"not-json"))
    assert sub.duplicate_count == 1 and len(seen) == 2
    assert audit.get_actionable_event_by_id("evt-1")["review_status"] == "CONFIRMED"


def test_redelivery_is_stored_when_first_insert_failed():
    db = MagicMock()
    db.insert_risk_event.side_effect = [RuntimeError("database is locked"), None]
    sub = MQTTEventSubscriber(audit_db=db)
    sub._client.on_message(sub._client, None, _msg(RISK, RISK_EVENT))
    sub._client.on_message(sub._client, None, _msg(RISK, RISK_EVENT))  # at-least-once redelivery
    assert db.insert_risk_event.call_count == 2 and sub.duplicate_count == 0
    sub._client.on_message(sub._client, None, _msg(RISK, RISK_EVENT))
    assert db.insert_risk_event.call_count == 2 and sub.duplicate_count == 1


def test_subscriber_survives_failing_handlers():
    bad_db = MagicMock()
    bad_db.insert_risk_event.side_effect = RuntimeError("disk full")
    sub = MQTTEventSubscriber(audit_db=bad_db, on_event_callback=MagicMock(side_effect=RuntimeError("boom")))
    sub._client.on_message(sub._client, None, _msg(RISK, {"event_id": "x"}))


def test_subscriber_subscribes_to_configured_topics():
    cfg = MQTTConfig()
    sub = MQTTEventSubscriber(config=cfg)
    client = MagicMock()
    sub._client.on_connect(client, None, None, 0)
    topics = [t for t, _ in client.subscribe.call_args[0][0]]
    assert topics == [cfg.topics.risk_events, cfg.topics.telemetry, cfg.topics.health, cfg.topics.heartbeat]


# ------------------------------------------------------ real broker (optional)
@pytest.mark.skipif(shutil.which("mosquitto") is None and not Path("/usr/sbin/mosquitto").exists(),
                    reason="mosquitto not installed")
def test_round_trip_through_real_broker_with_link_loss(tmp_path: Path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_spooler_resilience as bench

    port = bench.free_port()
    broker = bench.Broker(tmp_path, port)
    broker.start()
    try:
        res = bench.case_link_loss(port, tmp_path, outage_s=2.0)
    finally:
        broker.stop()
    assert res["missing_events"] == 0 and res["order_violations"] == 0 and res["peak_spool_depth"] > 0
