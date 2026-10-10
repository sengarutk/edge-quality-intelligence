#!/usr/bin/env python3
"""Per-stage latency of one inspection cycle, measured with time.perf_counter.

Stages per cycle (all executed for real):
  decode   PNG decode of a real nominal MVTec test image at native resolution (stands in for frame
           acquisition; camera driver and transport are not modeled)
  vision   optical health check + resize/normalize + PatchCore forward pass + score normalization
  sensor   SensorSimulator.step
  policy   TemporalPolicyEngine.evaluate
  spool    DiskSpooler insert of the decision (SQLite WAL, synchronous=NORMAL)
  audit    AuditLogDB insert of the decision
With ``persistence = async`` the spool and audit inserts run on a background writer thread
(src/runtime/async_writer.py); the inspection thread only enqueues the decision (stage
``persist``), and the writer's submit-to-commit lag is reported separately.
The end-to-end time is the wall time of the whole cycle on the inspection thread. Each device is
measured in both persistence modes. Results go to results/latency_benchmark_summary.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.audit_log import AuditLogDB  # noqa: E402
from src.runtime.async_writer import AsyncPersistenceWriter  # noqa: E402
from src.config import InferenceConfig, OpticalHealthConfig, PolicyConfig, SpoolerConfig, SystemConfig  # noqa: E402
from src.inference_service import InferenceEngine  # noqa: E402
from src.policy import TemporalPolicyEngine  # noqa: E402
from src.sensor_simulator import MachineState, SensorSimulator  # noqa: E402
from src.spooler import DiskSpooler  # noqa: E402

SYNC_STAGES = ("decode", "vision", "sensor", "policy", "spool", "audit")
ASYNC_STAGES = ("decode", "vision", "sensor", "policy", "persist")
DEADLINE_MS = 1000.0 / 30.0


def stats(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


def run(category: str, device: str, n_cycles: int, warmup: int, persistence: str = "sync") -> Dict:
    bank = np.load(PROJECT_ROOT / "results" / "score_bank" / f"{category}.npz")
    model_path = PROJECT_ROOT / "results" / "score_bank" / "models" / f"{category}_patchcore.pt"
    cfg = SystemConfig(
        optical_health=OpticalHealthConfig(blur_laplacian_threshold=float(bank["blur_threshold"])),
        inference=InferenceConfig(backend="patchcore", model_path=str(model_path),
                                  score_reference=float(bank["d_ref"]), device=device),
    )
    engine = InferenceEngine(config=cfg)
    # Nominal test images only: some defect types trip the blur check and would skip inference,
    # which would understate latency (see scripts/analyze_optical_check.py).
    nominal_paths = bank["test_paths"][bank["test_labels"] == 0]
    encoded = [np.frombuffer((PROJECT_ROOT / p).read_bytes(), np.uint8) for p in nominal_paths]
    native_shape = cv2.imdecode(encoded[0], cv2.IMREAD_COLOR).shape

    tmp = Path(tempfile.mkdtemp(prefix="latency_"))
    spool = DiskSpooler(config=SpoolerConfig(db_path=str(tmp / "spool.db"), max_spool_records=1_000_000))
    audit = AuditLogDB(db_path=str(tmp / "audit.db"))
    policy = TemporalPolicyEngine(config=PolicyConfig())
    sensor = SensorSimulator(seed=7)
    topic = "inspection/line1/risk"
    writer = AsyncPersistenceWriter(spool, audit, topic) if persistence == "async" else None
    stages = ASYNC_STAGES if writer else SYNC_STAGES

    times: Dict[str, List[float]] = {k: [] for k in stages + ("e2e",)}
    bypass = 0
    for i in range(warmup + n_cycles):
        t0 = time.perf_counter()
        frame = cv2.imdecode(encoded[i % len(encoded)], cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        inf = engine.run_inference(frame)
        t2 = time.perf_counter()
        bypass += int(not inf.optical_health.is_valid)
        reading = sensor.step(MachineState.RUNNING)
        t3 = time.perf_counter()
        decision = policy.evaluate(inf, reading)
        t4 = time.perf_counter()
        if writer:
            writer.submit(decision)
            marks = (t0, t1, t2, t3, t4, time.perf_counter())
        else:
            spool.enqueue(topic, json.dumps(decision.to_mqtt_payload()), qos=1)
            t5 = time.perf_counter()
            audit.insert_risk_event(decision)
            marks = (t0, t1, t2, t3, t4, t5, time.perf_counter())
        if i >= warmup:
            for k, a, b in zip(stages, marks[:-1], marks[1:]):
                times[k].append((b - a) * 1000.0)
            times["e2e"].append((marks[-1] - t0) * 1000.0)
        time.sleep(max(0.0, DEADLINE_MS / 1000.0 - (time.perf_counter() - t0)))  # paced at 30 FPS
    writer_stats = None
    if writer:
        writer.close()
        lag = writer.lag_ms[warmup:]
        writer_stats = {"lag": stats(lag), "max_queue_depth": writer.max_depth, "errors": writer.errors,
                        "persisted": spool.count_on_disk()}
    spool.close()
    audit.close()

    if bypass:
        raise RuntimeError(f"{bypass} cycles skipped inference (optical check failed); latency would be understated")
    e2e = np.asarray(times["e2e"])
    return {
        "category": category,
        "device": device,
        "persistence": persistence,
        "paced_fps": 30,
        "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor() or platform.machine(),
        "cycles": n_cycles,
        "warmup_cycles": warmup,
        "native_frame_shape": list(native_shape),
        "model_input": list(cfg.inference.input_resolution),
        "memory_bank_size": int(engine._patchcore.memory_bank.shape[0]),
        "optical_bypass_rate": bypass / (warmup + n_cycles),
        "deadline_ms": DEADLINE_MS,
        "deadline_miss_rate": float(np.mean(e2e > DEADLINE_MS)),
        "stages": {k: stats(v) for k, v in times.items()},
        "writer": writer_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="metal_nut")
    ap.add_argument("--gpu-cycles", type=int, default=5000)
    ap.add_argument("--cpu-cycles", type=int, default=500)
    args = ap.parse_args()
    logger.remove()

    runs = []
    for persistence in ("sync", "async"):
        if torch.cuda.is_available():
            runs.append(run(args.category, "cuda", args.gpu_cycles, warmup=100, persistence=persistence))
        runs.append(run(args.category, "cpu", args.cpu_cycles, warmup=20, persistence=persistence))
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "runs": runs,
    }
    out = PROJECT_ROOT / "results" / "latency_benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in runs:
        s = r["stages"]
        print(f"[{r['device']}/{r['persistence']}] e2e mean {s['e2e']['mean_ms']:.2f} ms, p95 {s['e2e']['p95_ms']:.2f} ms, "
              f"max {s['e2e']['max_ms']:.2f} ms, miss rate {r['deadline_miss_rate']:.4f}; "
              f"policy p95 {s['policy']['p95_ms']:.3f} ms; writer {r['writer']}")


if __name__ == "__main__":
    main()
