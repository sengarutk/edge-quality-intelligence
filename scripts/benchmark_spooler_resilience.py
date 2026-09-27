#!/usr/bin/env python3
"""Durability of the spool + MQTT path against a real Mosquitto broker.

A private Mosquitto instance is started on a free local port. A persistent-session
subscriber (QoS 1) records every delivery. The publisher (ResilientMQTTPublisher over a
DiskSpooler, SQLite WAL, synchronous=NORMAL) emits real PolicyDecision payloads at 30 Hz.

Fault cases
  link_loss_<D>s   the publisher's connection is closed for D seconds (pause/resume)
  broker_restart   the broker is stopped with SIGTERM for 30 s and restarted on the same
                   persistence directory
  publisher_crash  a publisher process builds a backlog offline and is killed with SIGKILL
                   while draining; a new process reopens the same spool file and finishes
  overflow         a 60 s outage with a 1,000-record spool, i.e. more events than capacity

For each case: generated, delivered (unique), missing, duplicate deliveries, order
violations among first deliveries, evicted records, peak spool depth and drain time
(reconnect until the spool is empty and every surviving event has been received).
Output: results/spooler_stress/spooler_stress_summary.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.config import MQTTBrokerConfig, MQTTConfig, SpoolerConfig  # noqa: E402
from src.inference_service import InferenceResult, OpticalHealthStatus  # noqa: E402
from src.mqtt_publisher import ResilientMQTTPublisher  # noqa: E402
from src.policy import TemporalPolicyEngine  # noqa: E402
from src.sensor_simulator import MachineState, SensorSimulator  # noqa: E402
from src.spooler import DiskSpooler  # noqa: E402

TOPIC = "inspection/line1/risk"
RATE_HZ = 30.0
MOSQUITTO = shutil.which("mosquitto") or "/usr/sbin/mosquitto"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Broker:
    def __init__(self, workdir: Path, port: int) -> None:
        self.port = port
        self.conf = workdir / "mosquitto.conf"
        (workdir / "data").mkdir(exist_ok=True)
        self.conf.write_text(
            f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence true\n"
            f"persistence_location {workdir / 'data'}/\nautosave_interval 1\nmax_queued_messages 0\n"
            "log_dest none\n"
        )
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.proc = subprocess.Popen([MOSQUITTO, "-c", str(self.conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("mosquitto did not start")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=10)


class Collector:
    """Persistent-session QoS 1 subscriber recording (event_id, sequence_id) per delivery."""

    def __init__(self, port: int) -> None:
        self.deliveries: List[tuple] = []
        self.lock = threading.Lock()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"collector_{uuid.uuid4().hex[:8]}",
                                  clean_session=False)
        self.client.on_connect = lambda c, u, f, rc, p=None: c.subscribe(TOPIC, qos=1)
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(1, 2)
        self.client.connect("127.0.0.1", port, keepalive=10)
        self.client.loop_start()

    def _on_message(self, client, userdata, msg) -> None:
        d = json.loads(msg.payload)
        with self.lock:
            self.deliveries.append((d["event_id"], d["sequence_id"]))

    def unique_count(self) -> int:
        with self.lock:
            return len({e for e, _ in self.deliveries})

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


def mqtt_config(port: int) -> MQTTConfig:
    return MQTTConfig(broker=MQTTBrokerConfig(host="127.0.0.1", port=port, keepalive=5,
                                              reconnect_delay_min_s=1.0, reconnect_delay_max_s=2.0))


class EventSource:
    """Real decisions from the policy engine on synthetic nominal inputs."""

    def __init__(self) -> None:
        self.engine = TemporalPolicyEngine(source_id="bench-gateway")
        self.sensor = SensorSimulator(seed=3)
        self.health = OpticalHealthStatus(is_valid=True, laplacian_var=500.0, mean_brightness=120.0)

    def next_payload(self) -> Dict[str, Any]:
        inf = InferenceResult(timestamp_utc="1970-01-01T00:00:00.000Z", camera_id="cam", vision_score=0.1,
                              is_blurred=False, is_occluded=False, optical_health=self.health, latency_ms=0.0)
        return self.engine.evaluate(inf, self.sensor.step(MachineState.RUNNING)).to_mqtt_payload()


def generate(pub: ResilientMQTTPublisher, src: EventSource, seconds: float, ids: List[str],
             on_tick=None, depth_log: Optional[List[int]] = None) -> None:
    period = 1.0 / RATE_HZ
    t0 = time.monotonic()
    k = 0
    while True:
        now = time.monotonic() - t0
        if now >= seconds:
            break
        if on_tick:
            on_tick(now)
        payload = src.next_payload()
        pub.publish_event(TOPIC, payload, qos=1)
        ids.append(payload["event_id"])
        if depth_log is not None:
            depth_log.append(pub.spooler.get_queue_depth())
        k += 1
        time.sleep(max(0.0, t0 + k * period - time.monotonic()))


def wait_drained(pub_spooler: DiskSpooler, col: Collector, expected: int, timeout: float = 120.0) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pub_spooler.count_on_disk() == 0 and col.unique_count() >= expected:
            return time.monotonic() - t0
        time.sleep(0.02)
    return float("nan")


def score(case: str, ids: List[str], col: Collector, evicted: int, peak: int, drain_s: float, **extra) -> Dict[str, Any]:
    time.sleep(1.0)  # let late duplicates arrive
    with col.lock:
        deliveries = list(col.deliveries)
    seen, first_seq = set(), []
    for eid, seq in deliveries:
        if eid not in seen:
            seen.add(eid)
            first_seq.append(seq)
    generated = set(ids)
    missing = len(generated - seen)
    return {
        "case": case,
        "events_generated": len(generated),
        "events_delivered_unique": len(generated & seen),
        "missing_events": missing,
        "evicted_by_capacity": evicted,
        "unexplained_missing": missing - evicted,
        "duplicate_deliveries": len(deliveries) - len(seen),
        "order_violations": sum(1 for a, b in zip(first_seq, first_seq[1:]) if b < a),
        "peak_spool_depth": peak,
        "drain_seconds": drain_s,
        **extra,
    }


def case_link_loss(port: int, work: Path, outage_s: float, capacity: int = 50_000, name: Optional[str] = None) -> Dict:
    col = Collector(port)
    spool = DiskSpooler(config=SpoolerConfig(db_path=str(work / f"{uuid.uuid4().hex}.db"), max_spool_records=capacity))
    pub = ResilientMQTTPublisher(config=mqtt_config(port), spooler=spool)
    pub.start()
    time.sleep(1.0)
    ids: List[str] = []
    depth: List[int] = []
    state = {"paused": False, "resumed": False}

    def tick(now: float) -> None:
        if not state["paused"] and now >= 5.0:
            pub.pause_network()
            state["paused"] = True
        if state["paused"] and not state["resumed"] and now >= 5.0 + outage_s:
            pub.resume_network()
            state["resumed"] = True

    src = EventSource()
    generate(pub, src, 5.0 + outage_s + 0.001, ids, tick, depth)
    if not state["resumed"]:
        pub.resume_network()
    drain = wait_drained(spool, col, len(ids) - spool.evicted_count)
    res = score(name or f"link_loss_{int(outage_s)}s", ids, col, spool.evicted_count, max(depth), drain,
                outage_seconds=outage_s, spool_capacity=capacity)
    pub.stop()
    spool.close()
    col.close()
    return res


def case_broker_restart(broker: Broker, port: int, work: Path, outage_s: float = 30.0) -> Dict:
    col = Collector(port)
    spool = DiskSpooler(config=SpoolerConfig(db_path=str(work / "restart.db"), max_spool_records=50_000))
    pub = ResilientMQTTPublisher(config=mqtt_config(port), spooler=spool)
    pub.start()
    time.sleep(1.0)
    ids: List[str] = []
    depth: List[int] = []
    state = {"down": False, "up": False}

    def tick(now: float) -> None:
        if not state["down"] and now >= 5.0:
            broker.stop()
            state["down"] = True
        if state["down"] and not state["up"] and now >= 5.0 + outage_s:
            broker.start()
            state["up"] = True

    generate(pub, EventSource(), 5.0 + outage_s + 0.001, ids, tick, depth)
    if not state["up"]:
        broker.start()
    drain = wait_drained(spool, col, len(ids))
    res = score("broker_restart_30s", ids, col, spool.evicted_count, max(depth), drain, outage_seconds=outage_s)
    pub.stop()
    spool.close()
    col.close()
    return res


def _crash_child(port: int, db: str, id_file: str, backlog_s: float, drain_only: bool) -> None:
    logger.remove()
    spool = DiskSpooler(config=SpoolerConfig(db_path=db, max_spool_records=50_000))
    pub = ResilientMQTTPublisher(config=mqtt_config(port), spooler=spool, max_inflight=20)
    if drain_only:
        pub.start()
        while True:
            time.sleep(1.0)
    ids: List[str] = []
    with open(id_file, "a", buffering=1) as fh:
        src = EventSource()
        period = 1.0 / RATE_HZ
        for k in range(int(backlog_s * RATE_HZ)):  # offline: everything goes to the spool
            payload = src.next_payload()
            pub.publish_event(TOPIC, payload, qos=1)
            fh.write(payload["event_id"] + "\n")
            time.sleep(period)
    pub.start()  # connect and drain; the parent kills this process mid-drain
    while True:
        time.sleep(1.0)


def case_publisher_crash(port: int, work: Path, backlog_s: float = 20.0) -> Dict:
    col = Collector(port)
    db, id_file = str(work / "crash.db"), str(work / "crash_ids.txt")
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_crash_child, args=(port, db, id_file, backlog_s, False))
    child.start()
    ids_expected = int(backlog_s * RATE_HZ)
    # Kill once draining has visibly started but is not complete.
    t0 = time.monotonic()
    while col.unique_count() < ids_expected // 4 and time.monotonic() - t0 < backlog_s + 30:
        time.sleep(0.001)
    delivered_before_kill = col.unique_count()
    os.kill(child.pid, signal.SIGKILL)
    child.join()
    peak = DiskSpooler(config=SpoolerConfig(db_path=db, max_spool_records=50_000)).count_on_disk()
    restart = ctx.Process(target=_crash_child, args=(port, db, id_file, 0.0, True))
    restart.start()
    ids = Path(id_file).read_text().split()
    check = DiskSpooler(config=SpoolerConfig(db_path=db, max_spool_records=50_000))
    drain = wait_drained(check, col, len(ids))
    res = score("publisher_crash_sigkill", ids, col, 0, peak, drain,
                delivered_before_kill=delivered_before_kill, spool_depth_after_kill=peak)
    restart.kill()
    restart.join()
    col.close()
    return res


def main() -> None:
    logger.remove()
    work = Path(tempfile.mkdtemp(prefix="spool_bench_"))
    port = free_port()
    broker = Broker(work, port)
    broker.start()
    results = []
    try:
        for outage in (30.0, 60.0, 120.0):
            results.append(case_link_loss(port, work, outage))
            print(json.dumps(results[-1]))
        results.append(case_broker_restart(broker, port, work))
        print(json.dumps(results[-1]))
        results.append(case_publisher_crash(port, work))
        print(json.dumps(results[-1]))
        results.append(case_link_loss(port, work, 60.0, capacity=1000, name="overflow_60s_capacity_1000"))
        print(json.dumps(results[-1]))
    finally:
        broker.stop()
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "broker": subprocess.run([MOSQUITTO, "-h"], capture_output=True, text=True).stdout.splitlines()[0],
        "event_rate_hz": RATE_HZ,
        "mqtt_qos": 1,
        "sqlite_journal_mode": "WAL",
        "sqlite_synchronous": "NORMAL",
        "cases": results,
    }
    out = PROJECT_ROOT / "results" / "spooler_stress" / "spooler_stress_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
