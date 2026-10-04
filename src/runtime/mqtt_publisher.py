"""Store-and-forward MQTT publisher backed by the local disk spool.

Delivery model
--------------
* QoS >= 1 events are written to the spool before any network I/O (write-ahead).
  A single drain thread sends spooled records in FIFO order and deletes a record
  only when paho reports the broker acknowledgement (PUBACK for QoS 1, PUBCOMP
  for QoS 2) through ``on_publish``. Delivery is therefore at-least-once:
  after a process crash a record whose acknowledgement was not yet processed is
  sent again, and consumers deduplicate on ``event_id``.
* QoS 0 messages (telemetry, heartbeats) are best effort: they are sent when the
  client is connected and counted as dropped otherwise.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict, Optional, Union

import paho.mqtt.client as mqtt
from loguru import logger

from src.config import MQTTConfig, load_mqtt_config
from src.spooler import DiskSpooler


class ResilientMQTTPublisher:
    """MQTT publisher with durable spooling and acknowledgement-driven deletion."""

    def __init__(
        self,
        config: Optional[MQTTConfig] = None,
        spooler: Optional[DiskSpooler] = None,
        max_inflight: int = 100,
        drain_interval_s: float = 0.05,
    ) -> None:
        self.config = config or load_mqtt_config()
        self.spooler = spooler or DiskSpooler(
            db_path=self.config.spooler.db_path,
            max_spool_records=self.config.spooler.max_spool_records,
        )
        self.client_id = self.config.broker.client_id or f"{self.config.broker.client_id_prefix}_{uuid.uuid4().hex[:12]}"
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            clean_session=self.config.broker.clean_session,
        )
        self._client.max_inflight_messages_set(max_inflight)
        self.max_inflight = max_inflight
        self.drain_interval_s = drain_interval_s

        self._is_connected = False
        self._network_paused = False
        self._running = False
        self._drain_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._inflight: Dict[int, int] = {}  # paho mid -> spool row id
        self._inflight_rows: set[int] = set()
        self._early_acks: set[int] = set()  # mids acknowledged before their mapping was recorded

        self.stats = {"spooled": 0, "deduplicated": 0, "acknowledged": 0, "qos0_sent": 0, "qos0_dropped": 0}
        self._setup_callbacks()
        logger.info(f"Initialized ResilientMQTTPublisher (client_id={self.client_id})")

    # ----------------------------------------------------------------- status
    @property
    def is_connected(self) -> bool:
        with self._state_lock:
            return self._is_connected and not self._network_paused

    def _setup_callbacks(self) -> None:
        def on_connect(client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
            code = getattr(rc, "value", rc)
            with self._state_lock:
                self._is_connected = code == 0
            if code == 0:
                logger.info(f"MQTT publisher connected to {self.config.broker.host}:{self.config.broker.port}")
            else:
                logger.warning(f"MQTT connection refused (rc={code})")

        def on_disconnect(client: mqtt.Client, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
            # The mid -> row mapping is kept on purpose. paho keeps unacknowledged QoS>=1
            # messages in its own outgoing queue and re-sends them (DUP) after every reconnect,
            # with or without clean_session, so their PUBACKs still arrive under the old mids.
            # Dropping the mapping here would turn those late acks into stale "early acks"
            # that could later delete an unrelated row once paho's mid counter wraps.
            with self._state_lock:
                self._is_connected = False
            logger.warning(f"MQTT publisher disconnected (rc={rc})")

        def on_publish(client: mqtt.Client, userdata: Any, mid: int, reason_code: Any = None, properties: Any = None) -> None:
            with self._state_lock:
                row_id = self._inflight.pop(mid, None)
                if row_id is None:
                    self._early_acks.add(mid)
                    return
                self._inflight_rows.discard(row_id)
            self._ack_row(row_id)

        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect
        self._client.on_publish = on_publish

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Connect asynchronously and start the network loop and drain thread."""
        with self._state_lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()

        self._client.reconnect_delay_set(
            min_delay=max(1, round(self.config.broker.reconnect_delay_min_s)),
            max_delay=max(1, round(self.config.broker.reconnect_delay_max_s)),
        )
        try:
            self._client.connect_async(
                host=self.config.broker.host, port=self.config.broker.port, keepalive=self.config.broker.keepalive
            )
            self._client.loop_start()
        except Exception as exc:  # the drain loop keeps spooling while offline
            logger.warning(f"Initial MQTT connect failed: {exc}. Events will be spooled.")

        self._drain_thread = threading.Thread(target=self._drain_worker, daemon=True, name="mqtt_spool_drainer")
        self._drain_thread.start()

    def stop(self, flush_timeout_s: float = 0.0) -> None:
        """Stop the drain thread and disconnect. Undelivered records stay in the spool."""
        with self._state_lock:
            if not self._running:
                return
        if flush_timeout_s > 0:
            self.flush(flush_timeout_s)
        with self._state_lock:
            self._running = False
        self._stop_event.set()
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=2.0)
        try:
            self._client.disconnect()
            self._client.loop_stop()
        except Exception as exc:
            logger.debug(f"Disconnect cleanup exception: {exc}")

    close = stop

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Block until the spool is empty (all records acknowledged) or the timeout expires."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.spooler.get_queue_depth() == 0:
                return True
            time.sleep(0.01)
        return self.spooler.get_queue_depth() == 0

    def pause_network(self) -> None:
        """Drop the broker connection on purpose (client-side partition used by fault injection)."""
        with self._state_lock:
            self._network_paused = True
        try:
            self._client.disconnect()
            self._client.loop_stop()
        except Exception as exc:
            logger.debug(f"pause_network: {exc}")

    def resume_network(self) -> None:
        """Re-establish the broker connection after pause_network()."""
        with self._state_lock:
            self._network_paused = False
        try:
            self._client.connect_async(
                host=self.config.broker.host, port=self.config.broker.port, keepalive=self.config.broker.keepalive
            )
            self._client.loop_start()
        except Exception as exc:
            logger.warning(f"resume_network: reconnect failed ({exc})")

    # ---------------------------------------------------------------- publish
    def _qos_for_topic(self, topic: str) -> int:
        t, q = self.config.topics, self.config.qos
        return {t.risk_events: q.risk_events, t.telemetry: q.telemetry, t.health: q.health, t.heartbeat: q.heartbeat}.get(
            topic, q.risk_events
        )

    def publish_event(self, topic: str, payload: Union[Dict[str, Any], str], qos: Optional[int] = None) -> bool:
        """Publish an event.

        Returns:
            For QoS >= 1: True once the record is durably spooled (False if it duplicates a queued event_id).
            For QoS 0: True if handed to the network layer, False if dropped while offline.
        """
        payload_str = payload if isinstance(payload, str) else json.dumps(payload)
        effective_qos = self._qos_for_topic(topic) if qos is None else int(qos)

        if effective_qos >= 1:
            inserted = self.spooler.enqueue(topic, payload_str, effective_qos)
            self.stats["spooled" if inserted else "deduplicated"] += 1
            return inserted

        if self.is_connected:
            info = self._client.publish(topic, payload_str, qos=0)
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                self.stats["qos0_sent"] += 1
                return True
        self.stats["qos0_dropped"] += 1
        return False

    publish = publish_event

    def publish_heartbeat(self, status: Dict[str, Any]) -> bool:
        return self.publish_event(self.config.topics.heartbeat, status)

    # ------------------------------------------------------------------ drain
    def _ack_row(self, row_id: int) -> None:
        # Called from both paho's network thread and the drain thread.
        self.spooler.delete_acknowledged([row_id])
        with self._state_lock:
            self.stats["acknowledged"] += 1

    def _drain_once(self) -> int:
        """Send the oldest spooled records not yet in flight. Returns the number handed to paho."""
        if not self.is_connected:
            return 0
        with self._state_lock:
            room = self.max_inflight - len(self._inflight)
            skip = set(self._inflight_rows)
        if room <= 0:
            return 0
        sent = 0
        for rec_id, topic, payload, qos in self.spooler.peek_batch(limit=room + len(skip)):
            if rec_id in skip:
                continue
            if not self.is_connected or self._stop_event.is_set():
                break
            # Never hold our lock while calling into paho (its network thread holds its own
            # callback lock while running on_publish, which takes our lock).
            info = self._client.publish(topic, payload, qos=qos)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                break
            with self._state_lock:
                acked_early = info.mid in self._early_acks
                if acked_early:
                    self._early_acks.discard(info.mid)
                else:
                    self._inflight[info.mid] = rec_id
                    self._inflight_rows.add(rec_id)
            if acked_early:
                self._ack_row(rec_id)
            sent += 1
            if sent >= room:
                break
        return sent

    def _drain_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._drain_once()
            except Exception as exc:
                logger.error(f"Spool drain error: {exc}")
            self._stop_event.wait(self.drain_interval_s)
