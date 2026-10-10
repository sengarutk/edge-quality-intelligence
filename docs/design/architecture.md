# Architecture

```
camera frame ──> optical check ──(valid)──> PatchCore ──> normalized score v ─┐
                      │ (invalid: score skipped)                               │
sensors ──> z-scores ──> composite score s ───────────────────────────────────┤
machine state ────────────────────────────────────────────────────────────────┤
                                                                               v
                                  TemporalPolicyEngine (smoothing, k-of-N, fusion,
                                  divergence triage, state gating, incident latch)
                                               │ PolicyDecision (every frame)
                     ┌─────────────────────────┼──────────────────────────┐
                     v                         v                          v
               AuditLogDB (SQLite)     DiskSpooler (SQLite WAL)    operator console
               (both inserted on the inspection thread, or by the optional writer thread)
                                               │ drain thread, FIFO      (Streamlit)
                                               v
                                   MQTT broker (QoS 1) ──> subscriber (dedup by event_id)
```

| Module | Role |
|---|---|
| `src/runtime/inference_service.py` | optical check, mock / PatchCore / ONNX scoring and normalization |
| `src/runtime/sensor_simulator.py` | simulated sensors and the shared `composite_sensor_score` |
| `src/runtime/policy.py` | decision policy and incident latch |
| `src/runtime/spooler.py` | durable FIFO spool with event-id dedup and counted eviction |
| `src/runtime/mqtt_publisher.py` | write-ahead publishing; rows deleted only after broker acknowledgement |
| `src/runtime/mqtt_subscriber.py` | ingestion into the audit log, duplicate suppression |
| `src/runtime/audit_log.py` | idempotent decision log and operator reviews |
| `src/runtime/async_writer.py` | optional writer thread for spool and audit inserts (decisions in its queue are lost on a crash) |
| `src/runtime/alarm_baselines.py` | external baselines: delay-timer, EMA with hysteresis, decision-level fusion |
| `src/runtime/fault_injector.py` | scheduled optical, sensor and network faults |
| `src/experiments/workloads.py`, `runtime_sim.py` | workload timelines and replay used by the studies |
| `src/metrics/stream.py`, `evaluator.py`, `queue_model.py` | alert metrics, aggregation, queue models |
