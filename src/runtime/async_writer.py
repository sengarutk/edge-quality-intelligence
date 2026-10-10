"""Background persistence of decisions (spool + audit log) off the inspection thread.

The inspection thread only hands the decision to an in-memory queue; one writer thread
serializes it, inserts it into the DiskSpooler (for MQTT delivery) and into the AuditLogDB.

Trade-off: a decision that is still in the queue when the process crashes is lost, whereas
the synchronous path (spool insert on the inspection thread) has it on disk before the next
frame. ``lag_ms`` records, per decision, the time from ``submit`` to the end of both inserts,
which bounds this exposure window. The queue is bounded; when it is full, ``submit`` blocks
(back-pressure) rather than dropping, and the blocked time is visible in the caller's latency.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import List, Optional

from src.audit_log import AuditLogDB
from src.policy import PolicyDecision
from src.spooler import DiskSpooler

_STOP = object()


class AsyncPersistenceWriter:
    def __init__(self, spooler: DiskSpooler, audit: Optional[AuditLogDB], topic: str,
                 max_queue: int = 10_000) -> None:
        self.spooler, self.audit, self.topic = spooler, audit, topic
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self.lag_ms: List[float] = []
        self.max_depth = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="persistence_writer")
        self._thread.start()

    def submit(self, decision: PolicyDecision) -> None:
        self._q.put((time.perf_counter(), decision))
        depth = self._q.qsize()
        if depth > self.max_depth:
            self.max_depth = depth

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                self._q.task_done()
                return
            t_submit, decision = item
            try:
                payload = decision.to_mqtt_payload()
                self.spooler.enqueue(self.topic, json.dumps(payload), qos=1)
                if self.audit is not None:
                    self.audit.insert_risk_event(payload)
            except Exception:  # counted; a failed insert must not stop the writer
                self.errors += 1
            self.lag_ms.append((time.perf_counter() - t_submit) * 1000.0)
            self._q.task_done()

    def close(self, timeout_s: float = 30.0) -> None:
        """Drain the queue and stop the writer thread."""
        self._q.put(_STOP)
        self._thread.join(timeout=timeout_s)
