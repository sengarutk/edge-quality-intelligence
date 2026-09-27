# Industrial Edge Messaging & Event Contracts

The runtime publishes to the topics configured in `configs/mqtt_config.yaml` (default prefix `inspection/line1/`). Payloads are produced by Pydantic models (`PolicyDecision.to_mqtt_payload()`, schema version 1.1).

---

## 1. Topic Hierarchy & QoS Matrix

| Topic Name | Direction | QoS Level | Spooled | Description |
| :--- | :---: | :---: | :---: | :--- |
| `inspection/line1/risk` | Publish / Subscribe | 1 (At least once) | Yes | Policy risk decisions and defect escalations |
| `inspection/line1/telemetry` | Publish / Subscribe | 0 (At most once) | No (dropped offline) | High-frequency physical sensor & vision scores |
| `inspection/line1/health` | Publish / Subscribe | 1 (At least once) | Yes | Subsystem health states and optical validation |
| `inspection/line1/heartbeat` | Publish | 0 (At most once) | No (dropped offline) | Runtime liveness heartbeats |

---

## 2. Event Payload Specifications

### 2.1 Risk Event (`inspection/line1/risk`)
Published every inspection cycle. `is_new_alert` marks decisions that enter the operator queue; `incident_id` groups the decisions of one incident. Delivery is at-least-once, so consumers must deduplicate on `event_id`.

```json
{
  "event_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "timestamp_utc": "2026-08-27T15:30:00.124Z",
  "camera_id": "line1_overhead_cam01",
  "machine_id": "press_unit_04",
  "machine_state": "RUNNING",
  "risk_state": "HIGH_SEVERITY",
  "trigger_reason": "SUSTAINED_VISION_ANOMALY",
  "raw_scores": {
    "vision_raw": 0.884,
    "sensor_raw": 0.052
  },
  "smoothed_scores": {
    "vision_ema": 0.821,
    "sensor_ema": 0.048
  },
  "window_stats": {
    "window_size_n": 10,
    "required_k": 4,
    "active_exceedances_count": {
      "vision_high": 4,
      "sensor_high": 0
    },
    "vision_confirmed_high": true,
    "sensor_confirmed": false
  },
  "cooldown_remaining": 15,
  "is_degraded": false,
  "incident_id": "2f6c0a51-7d0e-4f0f-9f5c-3b9c6c2f4a11",
  "is_new_alert": true,
  "source_id": "edge-gateway-01",
  "sequence_id": 1532,
  "created_monotonic_ns": 81234567890123,
  "schema_version": "1.1",
  "frame_id": "c3a81234-5678-4321-abcd-ef0123456789",
  "reading_id": "f4b98765-4321-1234-dcba-9876543210fe",
  "evidence_uri": "data/evidence/c3a81234-5678-4321-abcd-ef0123456789.png",
  "latency_ms": 9.42,
  "diagnostics": {
    "cross_modal_divergence": 0.773,
    "optical_degradation_reason": null,
    "missing_sensor_channels": [],
    "state_gated": false
  }
}
```

### 2.2 Telemetry Stream Event (`inspection/line1/telemetry`)
High-speed continuous time-series stream.

```json
{
  "timestamp_utc": "2026-08-27T15:30:00.124Z",
  "vibration_rms": 0.458,
  "temperature_c": 62.4,
  "current_amps": 12.85,
  "vision_score": 0.042,
  "sensor_score": 0.018
}
```

### 2.3 System Health Event (`inspection/line1/health`)
Component status updates and diagnostic alerts.

```json
{
  "timestamp_utc": "2026-08-27T15:30:00.124Z",
  "component": "camera_line1_overhead",
  "status": "HEALTHY",
  "details": "{\"laplacian_var\": 148.5, \"mean_brightness\": 127.8, \"fps\": 30.0}"
}
```