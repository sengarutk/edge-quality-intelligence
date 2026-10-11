#!/usr/bin/env python3
"""Failure cases beyond the publisher and broker: subscriber crashes, crashes of the inspection
process with the persistence writer thread, and the cost of SQLite synchronous=FULL.

Subscriber crash
  A real MQTTEventSubscriber with an AuditLogDB runs in a child process against a private
  Mosquitto broker. The publisher (ResilientMQTTPublisher over a DiskSpooler) emits real policy
  decisions at 30 Hz. After 5 s the subscriber is killed with SIGKILL and restarted D seconds
  later with the same configuration. Missing events are those never stored in the audit log.
  Cases: a random id with a clean session (the subscriber's behaviour before the
  subscriber_client_id option), and a fixed id with a persistent session, with the broker's
  default per-client queue (1,000 messages) and with an unlimited queue.

Inspection-process crash
  A child process runs the policy at 30 Hz and persists every decision either on the
  inspection thread (spool + audit insert before the next frame) or through the
  AsyncPersistenceWriter. After each hand-off (insert returned / submit returned) it writes the
  event id to its stdout pipe. The parent kills it with SIGKILL at a random time; a decision is
  lost if it was handed off but is not in the spool file afterwards.

Spool insert cost
  Unpaced spool inserts with synchronous=NORMAL and synchronous=FULL on the same disk. Power loss
  itself is not tested: NORMAL in WAL mode survives a process crash but may lose the last
  transactions on power loss, FULL does not, and this measures what FULL costs per insert.

Output: results/spooler_stress/delivery_failures_summary.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from scripts.benchmark_spooler_resilience import (  # noqa: E402
    MOSQUITTO, RATE_HZ, TOPIC, Broker, EventSource, free_port, mqtt_config,
)
from src.audit_log import AuditLogDB  # noqa: E402
from src.config import AuditConfig, MQTTBrokerConfig, MQTTConfig, SpoolerConfig  # noqa: E402
from src.mqtt_publisher import ResilientMQTTPublisher  # noqa: E402
from src.spooler import DiskSpooler  # noqa: E402

OUT = PROJECT_ROOT / "results" / "spooler_stress" / "delivery_failures_summary.json"


# ------------------------------------------------------------------ subscriber crash
def _subscriber_child(port: int, audit_db: str, client_id: Optional[str]) -> None:
    logger.remove()
    from src.mqtt_subscriber import MQTTEventSubscriber

    cfg = MQTTConfig(broker=MQTTBrokerConfig(host="127.0.0.1", port=port, keepalive=5, reconnect_delay_min_s=1.0,
                                             reconnect_delay_max_s=2.0, subscriber_client_id=client_id))
    sub = MQTTEventSubscriber(config=cfg, audit_db=AuditLogDB(config=AuditConfig(db_path=audit_db)))
    sub.start()
    while True:
        time.sleep(1.0)


def _stored_ids(audit_db: str) -> set:
    try:
        with sqlite3.connect(f"file:{audit_db}?mode=ro", uri=True, timeout=5) as con:
            return {r[0] for r in con.execute("SELECT event_id FROM risk_events;")}
    except sqlite3.OperationalError:
        return set()


def case_subscriber_crash(name: str, down_s: float, persistent: bool, max_queued: Optional[int]) -> Dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="subcrash_"))
    port = free_port()
    broker = Broker(work, port, max_queued_messages=max_queued)
    broker.start()
    audit_db = str(work / "audit.db")
    client_id = f"ingest_{uuid.uuid4().hex[:8]}" if persistent else None
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_subscriber_child, args=(port, audit_db, client_id))
    child.start()
    time.sleep(4.0)  # import, connect and subscribe

    spool = DiskSpooler(config=SpoolerConfig(db_path=str(work / "spool.db"), max_spool_records=50_000))
    pub = ResilientMQTTPublisher(config=mqtt_config(port), spooler=spool)
    pub.start()
    time.sleep(1.0)
    src, ids, during = EventSource(), [], []
    kill_at, restart_at, end = 5.0, 5.0 + down_s, 5.0 + down_s + 5.0
    killed = restarted = False
    period, t0, k = 1.0 / RATE_HZ, time.monotonic(), 0
    while (now := time.monotonic() - t0) < end:
        if not killed and now >= kill_at:
            os.kill(child.pid, signal.SIGKILL)
            child.join()
            killed = True
        if killed and not restarted and now >= restart_at:
            child = ctx.Process(target=_subscriber_child, args=(port, audit_db, client_id))
            child.start()
            restarted = True
        payload = src.next_payload()
        pub.publish_event(TOPIC, payload, qos=1)
        ids.append(payload["event_id"])
        if killed and not restarted:
            during.append(payload["event_id"])
        k += 1
        time.sleep(max(0.0, t0 + k * period - time.monotonic()))

    # Wait until every event is stored, or the stored set stops growing for 10 s.
    last, stable_since, deadline = -1, time.monotonic(), time.monotonic() + 120
    while time.monotonic() < deadline:
        n = len(_stored_ids(audit_db) & set(ids))
        if n == len(ids):
            break
        if n != last:
            last, stable_since = n, time.monotonic()
        elif time.monotonic() - stable_since > 10.0:
            break
        time.sleep(0.25)
    stored = _stored_ids(audit_db) & set(ids)
    publisher_backlog = spool.count_on_disk()
    child.kill()
    child.join()
    pub.stop()
    spool.close()
    broker.stop()
    shutil.rmtree(work, ignore_errors=True)
    missing = set(ids) - stored
    return {
        "case": name,
        "subscriber_session": "persistent (fixed id)" if persistent else "clean (random id)",
        "broker_max_queued_messages": "default (1000)" if max_queued is None else ("unlimited" if max_queued == 0 else max_queued),
        "subscriber_down_seconds": down_s,
        "events_generated": len(ids),
        "events_generated_while_down": len(during),
        "events_stored": len(stored),
        "missing_events": len(missing),
        "missing_generated_while_down": len(missing & set(during)),
        "publisher_spool_after_run": publisher_backlog,
    }


# ------------------------------------------------------------- inspection crash
def inspection_child(mode: str, spool_db: str, audit_db: str) -> None:
    """Run the policy at 30 Hz forever; report each event id once it has been handed off."""
    logger.remove()
    from src.runtime.async_writer import AsyncPersistenceWriter
    from src.inference_service import InferenceResult, OpticalHealthStatus
    from src.policy import TemporalPolicyEngine
    from src.sensor_simulator import MachineState, SensorSimulator

    spool = DiskSpooler(config=SpoolerConfig(db_path=spool_db, max_spool_records=1_000_000))
    audit = AuditLogDB(config=AuditConfig(db_path=audit_db))
    writer = AsyncPersistenceWriter(spool, audit, TOPIC) if mode == "writer_thread" else None
    engine, sensor = TemporalPolicyEngine(source_id="crash-gw"), SensorSimulator(seed=5)
    health = OpticalHealthStatus(is_valid=True, laplacian_var=500.0, mean_brightness=120.0)
    os.write(1, b"READY\n")
    period, t0, k = 1.0 / RATE_HZ, time.monotonic(), 0
    while True:
        inf = InferenceResult(timestamp_utc="1970-01-01T00:00:00.000Z", camera_id="cam", vision_score=0.1,
                              is_blurred=False, is_occluded=False, optical_health=health, latency_ms=0.0)
        decision = engine.evaluate(inf, sensor.step(MachineState.RUNNING))
        if writer is not None:
            writer.submit(decision)
            eid = decision.to_mqtt_payload()["event_id"]
        else:
            payload = decision.to_mqtt_payload()
            spool.enqueue(TOPIC, json.dumps(payload), qos=1)
            audit.insert_risk_event(payload)
            eid = payload["event_id"]
        os.write(1, (eid + "\n").encode())
        k += 1
        time.sleep(max(0.0, t0 + k * period - time.monotonic()))


def case_inspection_crash(mode: str, kills: int, rng: random.Random) -> Dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="inspcrash_"))
    per_kill, handed_off = [], 0
    for i in range(kills):
        spool_db, audit_db = str(work / f"s{i}.db"), str(work / f"a{i}.db")
        proc = subprocess.Popen([sys.executable, __file__, "--child", mode, spool_db, audit_db],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=PROJECT_ROOT)
        assert proc.stdout.readline().strip() == b"READY"
        time.sleep(rng.uniform(1.0, 3.0))
        proc.send_signal(signal.SIGKILL)
        out, _ = proc.communicate()
        reported = [ln for ln in out.decode().split() if ln]
        with sqlite3.connect(spool_db) as con:
            spooled = {r[0] for r in con.execute("SELECT event_id FROM spool_queue;")}
        lost = [e for e in reported if e not in spooled]
        per_kill.append(len(lost))
        handed_off += len(reported)
    shutil.rmtree(work, ignore_errors=True)
    return {
        "mode": mode,
        "kills": kills,
        "decisions_handed_off": handed_off,
        "decisions_lost": int(sum(per_kill)),
        "kills_with_loss": int(sum(1 for n in per_kill if n)),
        "max_lost_per_kill": int(max(per_kill)),
    }


# ------------------------------------------------------------------ insert cost
def insert_cost(n: int = 2000) -> List[Dict[str, Any]]:
    work = Path(tempfile.mkdtemp(prefix="synccost_", dir=str(PROJECT_ROOT / "data")))
    src, rows = EventSource(), []
    payloads = [json.dumps(src.next_payload()) for _ in range(n)]
    for sync in ("NORMAL", "FULL"):
        sp = DiskSpooler(config=SpoolerConfig(db_path=str(work / f"{sync}.db"), max_spool_records=1_000_000),
                         synchronous=sync)
        ms = []
        for p in payloads:
            t = time.perf_counter()
            sp.enqueue(TOPIC, p, qos=1)
            ms.append((time.perf_counter() - t) * 1000.0)
        sp.close()
        a = np.array(ms)
        rows.append({"synchronous": sync, "inserts": n, "median_ms": float(np.median(a)),
                     "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())})
    shutil.rmtree(work, ignore_errors=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kills", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logger.remove()
    rng = random.Random(args.seed)
    sub_cases = []
    for name, down, persistent, queue in (
        ("clean_session_10s", 10.0, False, None),
        ("persistent_10s_default_queue", 10.0, True, None),
        ("persistent_60s_default_queue", 60.0, True, None),
        ("persistent_60s_unlimited_queue", 60.0, True, 0),
    ):
        sub_cases.append(case_subscriber_crash(name, down, persistent, queue))
        print(json.dumps(sub_cases[-1]), flush=True)
    crash_cases = []
    for mode in ("synchronous", "writer_thread"):
        crash_cases.append(case_inspection_crash(mode, args.kills, rng))
        print(json.dumps(crash_cases[-1]), flush=True)
    cost = insert_cost()
    print(json.dumps(cost), flush=True)
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "broker": subprocess.run([MOSQUITTO, "-h"], capture_output=True, text=True).stdout.splitlines()[0],
        "event_rate_hz": RATE_HZ,
        "seed": args.seed,
        "subscriber_crash": sub_cases,
        "inspection_crash": crash_cases,
        "spool_insert_cost": cost,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        inspection_child(*sys.argv[2:])
    else:
        main()
