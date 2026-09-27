"""Local SQLite spool used while the MQTT broker is unreachable.

Records are appended in arrival order and removed only after the broker has
acknowledged them (see ResilientMQTTPublisher). Capacity is bounded; when it is
exceeded the oldest records are evicted and counted, so data loss is never silent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from src.config import SpoolerConfig, load_mqtt_config


class DiskSpooler:
    """Thread-safe, file-backed FIFO queue with event-id deduplication."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_spool_records: Optional[int] = None,
        config: Optional[SpoolerConfig] = None,
        synchronous: str = "NORMAL",
    ) -> None:
        """Open (or create) the spool database.

        Args:
            db_path: Path of the SQLite file.
            max_spool_records: Capacity before FIFO eviction.
            config: Optional SpoolerConfig (takes precedence over the two arguments above).
            synchronous: SQLite ``synchronous`` pragma (``NORMAL`` or ``FULL``).
        """
        if config is not None:
            self.db_path = Path(config.db_path)
            self.max_records = config.max_spool_records
        else:
            mqtt_cfg = load_mqtt_config()
            self.db_path = Path(db_path or mqtt_cfg.spooler.db_path)
            self.max_records = max_spool_records or mqtt_cfg.spooler.max_spool_records
        if synchronous.upper() not in ("NORMAL", "FULL"):
            raise ValueError("synchronous must be NORMAL or FULL")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute(f"PRAGMA synchronous = {synchronous.upper()};")
        self.synchronous = synchronous.upper()
        self._init_db()
        self._depth = self._count()
        self.evicted_count = 0
        self.duplicate_count = 0

        logger.info(f"Initialized DiskSpooler (db={self.db_path}, max_records={self.max_records})")

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spool_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    qos INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0
                );
                """
            )
            cols = [c[1] for c in self._conn.execute("PRAGMA table_info(spool_queue);").fetchall()]
            if "event_id" not in cols:
                self._conn.execute("ALTER TABLE spool_queue ADD COLUMN event_id TEXT;")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_spool_event_id ON spool_queue(event_id) "
                "WHERE event_id IS NOT NULL;"
            )

    def _count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM spool_queue;").fetchone()[0])

    @staticmethod
    def _extract_event_id(payload: str) -> Optional[str]:
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, dict):
            eid = parsed.get("event_id") or parsed.get("decision_id")
            return str(eid) if eid else None
        return None

    def enqueue(self, topic: str, payload: str, qos: int = 1) -> bool:
        """Append a record.

        Returns:
            True if a new row was written, False if a record with the same event_id was already queued.
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        event_id = self._extract_event_id(payload)

        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO spool_queue (event_id, topic, payload, qos, created_at) VALUES (?, ?, ?, ?, ?);",
                (event_id, topic, payload, int(qos), now_utc),
            )
            inserted = cur.rowcount == 1
            if not inserted:
                self.duplicate_count += 1
                return False
            self._depth += 1
            if self._depth > self.max_records:
                self._evict(self._depth - self.max_records)
        return True

    def _evict(self, n: int) -> int:
        cur = self._conn.execute(
            "DELETE FROM spool_queue WHERE id IN (SELECT id FROM spool_queue ORDER BY id ASC LIMIT ?);", (n,)
        )
        removed = cur.rowcount
        self._depth -= removed
        self.evicted_count += removed
        logger.warning(f"DiskSpooler at capacity: evicted {removed} oldest record(s) (total evicted {self.evicted_count}).")
        return removed

    def peek_batch(self, limit: int = 50) -> List[Tuple[int, str, str, int]]:
        """Oldest records first: (row_id, topic, payload, qos)."""
        with self._lock:
            return self._conn.execute(
                "SELECT id, topic, payload, qos FROM spool_queue ORDER BY id ASC LIMIT ?;", (limit,)
            ).fetchall()

    def mark_retry(self, record_ids: List[int]) -> None:
        """Increment the retry counter of records whose delivery attempt did not complete."""
        if not record_ids:
            return
        placeholders = ",".join("?" for _ in record_ids)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE spool_queue SET retry_count = retry_count + 1 WHERE id IN ({placeholders});", record_ids
            )

    def delete_acknowledged(self, record_ids: List[int]) -> int:
        """Remove records whose delivery was acknowledged by the broker."""
        if not record_ids:
            return 0
        placeholders = ",".join("?" for _ in record_ids)
        with self._lock, self._conn:
            cur = self._conn.execute(f"DELETE FROM spool_queue WHERE id IN ({placeholders});", record_ids)
            self._depth -= cur.rowcount
            return cur.rowcount

    def count_on_disk(self) -> int:
        """Exact number of queued rows read from the database (sees writes by other processes)."""
        return self._count()

    def get_queue_depth(self) -> int:
        """Number of records waiting for delivery."""
        with self._lock:
            return self._depth

    def purge_expired(self, max_records: int) -> int:
        """Evict oldest records until at most ``max_records`` remain."""
        with self._lock, self._conn:
            excess = self._depth - max_records
            return self._evict(excess) if excess > 0 else 0

    def clear(self) -> int:
        """Delete every queued record (operator action; the count is returned and logged)."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM spool_queue;")
            self._depth = 0
            logger.warning(f"DiskSpooler cleared by operator: {cur.rowcount} record(s) discarded.")
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            logger.info("DiskSpooler database connection closed.")
