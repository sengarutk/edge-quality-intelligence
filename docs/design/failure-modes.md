# Failure Modes and Runtime Responses

| Failure | Detection | Runtime response (FULL_POLICY) | Evidence |
|---|---|---|---|
| Camera blur | Variance of Laplacian at 224x224 below a per-category threshold (half the 5th percentile of held-out good frames) | Frame skips the detector and does not update the visual filter. 1-3 frames: held silently. k of the last N frames: one REVIEW alert (`OPTICAL_DEGRADATION_FALLBACK`). | `tests/test_rt_policy.py`, degraded-inputs workload |
| Dark or saturated frame | Mean brightness < 15 or > 245 | Same as blur | same |
| Defect mistaken for blur | Not detected | Some real defects (metal_nut "flip", cable "missing_cable") fall below the blur threshold and are routed as camera problems. | `results/score_bank/optical_check_on_test.json` |
| Sensor channel dropout | Channel listed in `missing_channels` | Held value is marked degraded; the sensor filter is frozen; strong visual evidence goes to REVIEW (`SENSOR_DEGRADATION_FALLBACK`) instead of HIGH. | `tests/test_rt_policy.py` |
| Temperature measurement drift | Not distinguishable from real heating | Produces `SUSTAINED_SENSOR_ANOMALY` review alerts. | degraded-inputs workload |
| Glare / reflection | Not detected as an optical fault (focus stays high) | Filtered by persistence; if it lasts, routed to REVIEW by divergence triage, not to HIGH. | glare and multi-modal workloads |
| Broker outage or network loss | paho disconnect | Events stay in the SQLite spool (written before any network I/O) and are sent in order after reconnection. | `results/spooler_stress/` |
| Publisher crash | - | Unacknowledged spool rows survive and are re-sent by the next process (possible duplicates, removed by event_id). | `publisher_crash_sigkill` case |
| Spool full | Row count reaches `max_spool_records` | Oldest rows are evicted and counted (`DiskSpooler.evicted_count`); loss is logged, never silent. | `overflow_60s_capacity_1000` case |
| Power loss | - | WAL with `synchronous=NORMAL` may lose the most recent transactions. Use `synchronous=FULL` if this matters. | not tested |
