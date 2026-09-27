from __future__ import annotations

import sys
from pathlib import Path

# Guarantee project root is in sys.path when executed via `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__doc__ = """Industrial Edge Inspection Runtime - Operator Reliability & Triage Console."""

import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.audit_log import AuditLogDB
from src.config import (
    AuditConfig,
    MQTTConfig,
    PolicyConfig,
    SensorConfig,
    SpoolerConfig,
    SystemConfig,
    load_mqtt_config,
    load_policy_config,
    load_sensor_config,
    load_system_config,
)
from src.evidence_manager import EvidenceManager
from src.inference_service import InferenceEngine, InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorReading, SensorSimulator
from src.spooler import DiskSpooler


# =============================================================================
# Streamlit App Configuration & Styling
# =============================================================================
st.set_page_config(
    page_title="Edge Inspection Runtime | Operator Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    a.header-anchor { display: none !important; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2e344e;
        margin-bottom: 10px;
    }
    .status-badge-healthy {
        background-color: #0e4429;
        color: #3fb950;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-warning {
        background-color: #4d2d00;
        color: #d29922;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-critical {
        background-color: #4c1114;
        color: #f85149;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached Singletons & Database Access
# =============================================================================
@st.cache_resource
def get_database(db_path: Optional[str] = None) -> AuditLogDB:
    """Retrieve thread-safe AuditLogDB singleton connection."""
    if db_path:
        return AuditLogDB(db_path=db_path)
    return AuditLogDB()


@st.cache_resource
def get_evidence_mgr(storage_dir: Optional[str] = None) -> EvidenceManager:
    """Retrieve EvidenceManager singleton."""
    return EvidenceManager(storage_dir=storage_dir or "data/evidence")


def get_system_status(recent_events: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Derive global operational health status and styling badge."""
    if not recent_events:
        return "OPERATIONAL", "status-badge-healthy"

    latest = recent_events[0]
    if latest.get("risk_state") == "HIGH_SEVERITY":
        return "CRITICAL LOCKOUT", "status-badge-critical"
    if latest.get("risk_state") == "REVIEW_REQUIRED" or latest.get("is_degraded"):
        return "DEGRADED REVIEW", "status-badge-warning"

    return "OPERATIONAL", "status-badge-healthy"


# =============================================================================
# Realistic Industrial Texture & Heatmap Generation
# =============================================================================
def generate_synthetic_industrial_frame(
    width: int = 224, height: int = 224, seed: Optional[int] = None
) -> np.ndarray:
    """Generate a realistic brushed-steel texture with subtle weld lines."""
    rng = np.random.RandomState(seed or 42)
    x = np.linspace(110, 140, width)
    base = np.tile(x, (height, 1)).astype(np.float32)
    streaks = rng.normal(0, 8, (height, 1)).repeat(width, axis=1)
    grain = rng.normal(0, 4, (height, width))
    img = np.clip(base + streaks + grain, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(img_bgr, (width // 2, 0), (width // 2, height), (95, 95, 105), 2)
    return img_bgr


def generate_defect_heatmap(
    width: int = 224,
    height: int = 224,
    center: Tuple[int, int] = (112, 112),
    radius: int = 35,
) -> np.ndarray:
    """Generate a 2D Gaussian activation hotspot simulating deep PatchCore patch anomalies."""
    y, x = np.ogrid[:height, :width]
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * (radius**2))).astype(np.float32)
    return heatmap


# =============================================================================
# Interactive Demo Seeder & Chaos Simulation Helpers
# =============================================================================
def seed_demo_simulation(
    n_steps: int = 60,
    db: Optional[AuditLogDB] = None,
    evidence_mgr: Optional[EvidenceManager] = None,
) -> int:
    """Demo only: run a 60-step synthetic sequence (mock vision backend, simulated sensors)."""
    audit_db = db or get_database()
    ev_mgr = evidence_mgr or get_evidence_mgr()

    sensor_sim = SensorSimulator(seed=42)
    inf_engine = InferenceEngine(seed=42)
    policy_engine = TemporalPolicyEngine()

    base_time = datetime.now(timezone.utc) - timedelta(seconds=n_steps * 0.03333)
    events_generated = 0

    for step in range(n_steps):
        step_time = base_time + timedelta(milliseconds=step * 33.333)
        now_iso = step_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        is_defect = (20 <= step <= 25) or (42 <= step <= 45)
        is_sensor_fault = (32 <= step <= 36)
        is_optical_fault = (step == 52)

        frame = generate_synthetic_industrial_frame(width=224, height=224, seed=42 + step)
        if is_optical_fault:
            frame = cv2.GaussianBlur(frame, (31, 31), 0)

        inf_result = inf_engine.run_inference(frame, inject_anomaly=is_defect)
        inf_result.timestamp_utc = now_iso

        dropouts = ["current"] if (12 <= step <= 14) else []
        sensor_reading = sensor_sim.step(
            machine_state=MachineState.RUNNING,
            inject_fault=is_sensor_fault,
            simulate_dropout=dropouts,
        )
        sensor_reading.timestamp_utc = now_iso

        ev_uri = None
        if is_defect or is_optical_fault or is_sensor_fault:
            heatmap = inf_result.heatmap  # None for degraded frames: the raw frame is archived instead
            ev_uri = ev_mgr.save_evidence(frame, heatmap, f"sim_frame_{step:04d}")

        decision = policy_engine.evaluate(inf_result, sensor_reading, evidence_uri=ev_uri)
        decision.timestamp_utc = now_iso

        audit_db.insert_telemetry(sensor_reading, inf_result)
        audit_db.insert_risk_event(decision)
        events_generated += 1

    return events_generated


def inject_chaos_fault_event(
    fault_type: str,
    db: AuditLogDB,
    evidence_mgr: EvidenceManager,
) -> None:
    """Inject an immediate single-cycle anomalous condition and update audit records."""
    sensor_sim = SensorSimulator()
    inf_engine = InferenceEngine()
    policy_engine = TemporalPolicyEngine()

    frame = generate_synthetic_industrial_frame(seed=int(time.time() * 1000) % 100000)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    if fault_type == "OPTICAL_BLUR":
        frame = cv2.GaussianBlur(frame, (45, 45), 0)
        inf_result = inf_engine.run_inference(frame, inject_anomaly=False)
        inf_result.timestamp_utc = now_iso
        sensor_reading = sensor_sim.step(machine_state=MachineState.RUNNING)
        sensor_reading.timestamp_utc = now_iso
        ev_uri = evidence_mgr.save_evidence(
            frame, np.zeros((224, 224), dtype=np.float32), f"chaos_blur_{int(time.time())}"
        )
        decision = policy_engine.evaluate(inf_result, sensor_reading, evidence_uri=ev_uri)
        decision.timestamp_utc = now_iso
        db.insert_telemetry(sensor_reading, inf_result)
        db.insert_risk_event(decision)
        db.insert_system_health("camera_optics", "DEGRADED", "OPTICAL_BLUR_DETECTED: Laplacian Var < 100.0")

    elif fault_type == "NETWORK_PARTITION":
        # The console does not own a broker connection; it records the operator-declared state only.
        db.insert_system_health("mqtt_broker", "OFFLINE", "Operator-declared network partition (demo)")

    elif fault_type == "SENSOR_DRIFT":
        inf_result = inf_engine.run_inference(frame, inject_anomaly=False)
        inf_result.timestamp_utc = now_iso
        drift_c = 12.0
        sensor_reading = sensor_sim.step(machine_state=MachineState.RUNNING, temperature_offset_c=drift_c)
        sensor_reading.timestamp_utc = now_iso
        decision = policy_engine.evaluate(inf_result, sensor_reading)
        decision.timestamp_utc = now_iso
        db.insert_telemetry(sensor_reading, inf_result)
        db.insert_risk_event(decision)
        db.insert_system_health(
            "physical_sensors", "WARNING",
            f"Injected temperature drift +{drift_c:.0f} C (z = {sensor_reading.sensor_breakdown['temperature_zscore']:.1f})",
        )

    elif fault_type == "SENSOR_DROPOUT":
        inf_result = inf_engine.run_inference(frame, inject_anomaly=False)
        inf_result.timestamp_utc = now_iso
        sensor_reading = sensor_sim.step(
            machine_state=MachineState.RUNNING, simulate_dropout=["current"]
        )
        sensor_reading.timestamp_utc = now_iso
        decision = policy_engine.evaluate(inf_result, sensor_reading)
        decision.timestamp_utc = now_iso
        db.insert_telemetry(sensor_reading, inf_result)
        db.insert_risk_event(decision)
        db.insert_system_health("current_sensor", "DEGRADED", "Channel Dropout: Imputed ZOH")


def restore_system_nominal(db: AuditLogDB) -> None:
    """Restore all hardware and network subsystem states to nominal operating conditions."""
    db.insert_system_health("camera_optics", "HEALTHY", "Nominal 30 FPS / Laplacian Var >= 100.0")
    db.insert_system_health("patchcore_model", "HEALTHY", "Operator-declared nominal state")
    db.insert_system_health("physical_sensors", "HEALTHY", "Vibration, temperature and current channels reporting")
    db.insert_system_health("current_sensor", "HEALTHY", "Current channel reporting")
    db.insert_system_health("mqtt_broker", "CONNECTED", "Operator-declared nominal state")



def compute_dynamic_fmea(
    audit_db: AuditLogDB,
    recent_events: List[Dict[str, Any]],
    recent_telemetry: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Derive real-time dynamic FMEA subsystem health rows based on latest health status."""
    try:
        recent_health = audit_db.query_recent_health(limit=50)
    except Exception:
        recent_health = []

    health_by_comp: Dict[str, Dict[str, Any]] = {}
    for h in recent_health:
        comp = h.get("component")
        if comp and comp not in health_by_comp:
            health_by_comp[comp] = h

    cam_health = health_by_comp.get("camera_optics", {})
    if cam_health.get("status") == "HEALTHY":
        has_optical_degradation = False
    elif cam_health.get("status") == "DEGRADED":
        has_optical_degradation = True
    else:
        has_optical_degradation = any(
            e.get("is_degraded") or e.get("trigger_reason") == "OPTICAL_DEGRADATION_FALLBACK"
            for e in recent_events[:10]
        )

    camera_row = {
        "Subsystem": "Camera Optics (Line 1 Overhead)",
        "Health Status": "DEGRADED (Blur Detected)" if has_optical_degradation else "HEALTHY",
        "Degradation Mode": "Laplacian Var < 100.0" if has_optical_degradation else "Laplacian Var >= 100.0",
        "Action / Fallback": "Fallback to Sensor Telemetry" if has_optical_degradation else "Nominal",
    }

    model_health = health_by_comp.get("patchcore_model", {})
    latencies = [t["latency_ms"] for t in recent_telemetry[:20] if t.get("latency_ms") is not None]
    avg_latency = float(np.mean(latencies)) if latencies else float("nan")

    if model_health.get("status") == "HEALTHY":
        model_status = "HEALTHY"
        model_action = "Nominal"
    elif model_health.get("status") == "DEGRADED" or avg_latency > 25.0:
        model_status = "WARNING (High Latency)"
        model_action = "Throttle Frame Rate"
    else:
        model_status = "HEALTHY"
        model_action = "Nominal"

    model_row = {
        "Subsystem": "PatchCore Vision Inference Model",
        "Health Status": model_status,
        "Degradation Mode": f"Mean latency: {avg_latency:.1f} ms" if latencies else "No latency samples",
        "Action / Fallback": model_action,
    }

    sensor_health = health_by_comp.get("physical_sensors", {})
    current_health = health_by_comp.get("current_sensor", {})

    if sensor_health.get("status") == "HEALTHY" and current_health.get("status") == "HEALTHY":
        sensor_status = "HEALTHY"
        sensor_mode = "Continuous Sampling @ 30Hz"
        sensor_action = "Nominal"
    elif current_health.get("status") == "DEGRADED":
        sensor_status = "DEGRADED (Imputed ZOH)"
        sensor_mode = "Channel Dropout / Signal Detached"
        sensor_action = "Zero-Order Hold (ZOH) Fallback"
    elif sensor_health.get("status") == "WARNING":
        sensor_status = "WARNING (Elevated Load/Heat)"
        sensor_mode = "Excess Z-Score > 3.0"
        sensor_action = "Cross-Modal Anomaly Verification"
    else:
        has_dropout = any(
            t.get("vibration_rms", 0.0) == 0.0 or t.get("current_amps", 0.0) == 0.0
            for t in recent_telemetry[:10]
        )
        has_sensor_spike = any(
            t.get("sensor_score", 0.0) > 0.65 or t.get("temperature_c", 0.0) > 85.0
            for t in recent_telemetry[:10]
        )
        if has_dropout:
            sensor_status = "DEGRADED (Imputed ZOH)"
            sensor_mode = "Channel Dropout / Signal Detached"
            sensor_action = "Zero-Order Hold (ZOH) Fallback"
        elif has_sensor_spike:
            sensor_status = "WARNING (Elevated Load/Heat)"
            sensor_mode = "Excess Z-Score > 3.0"
            sensor_action = "Cross-Modal Anomaly Verification"
        else:
            sensor_status = "HEALTHY"
            sensor_mode = "Continuous Sampling @ 30Hz"
            sensor_action = "Nominal"

    sensor_row = {
        "Subsystem": "3-Axis Vibration & Thermal Sensors",
        "Health Status": sensor_status,
        "Degradation Mode": sensor_mode,
        "Action / Fallback": sensor_action,
    }

    mqtt_health = health_by_comp.get("mqtt_broker", {})
    if mqtt_health.get("status") in ("CONNECTED", "HEALTHY"):
        is_broker_offline = False
    elif mqtt_health.get("status") == "OFFLINE":
        is_broker_offline = True
    else:
        is_broker_offline = False

    mqtt_row = {
        "Subsystem": "MQTT broker",
        "Health Status": "DISCONNECTED (Offline)" if is_broker_offline else "CONNECTED",
        "Degradation Mode": "Broker unreachable" if is_broker_offline else "Connected",
        "Action / Fallback": "Disk Spooler Active" if is_broker_offline else "Nominal",
    }

    spooler = DiskSpooler()
    queue_depth = spooler.count_on_disk()
    spooler_capacity = spooler.max_records
    spooler.close()

    spooler_row = {
        "Subsystem": "SQLite Local Disk Spooler",
        "Health Status": f"BUFFERING (Active: {queue_depth} spooled)" if queue_depth > 0 else "HEALTHY (Idle)",
        "Degradation Mode": f"Queued: {queue_depth}/{spooler_capacity} records (evictions beyond capacity)",
        "Action / Fallback": "Store-and-forward until broker acknowledgement",
    }

    return [camera_row, model_row, sensor_row, mqtt_row, spooler_row]


# =============================================================================
# Main Application Flow
# =============================================================================
def main() -> None:
    """Render operator reliability dashboard UI components."""
    audit_db = get_database()
    evidence_mgr = get_evidence_mgr()

    recent_events = audit_db.query_recent_events(limit=50)
    recent_telemetry = audit_db.query_recent_telemetry(limit=60)
    operator_metrics = audit_db.get_operator_metrics()

    sys_status, status_class = get_system_status(recent_events)

    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.title("Industrial Edge Inspection Runtime")
        st.caption("Operator Reliability Console, Temporal Anomaly Fusion & Human-in-the-Loop Triage")
    with h_col2:
        st.markdown(
            f"<div style='text-align: right; padding-top: 15px;'>"
            f"System State: <span class='{status_class}'>{sys_status}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.title("Runtime Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Live Simulation Engine")
    if st.sidebar.button("⚡ Run 60-Cycle Live Simulation", width="stretch"):
        with st.spinner("Executing 60-cycle edge inspection simulation..."):
            count = seed_demo_simulation(60, audit_db, evidence_mgr)
        st.sidebar.success(f"Generated {count} telemetry cycles & events!")
        st.rerun()

    confirm_reset = st.sidebar.checkbox("I understand the reset deletes all audit records")
    if st.sidebar.button("🧹 Reset & Clean Database", width="stretch", disabled=not confirm_reset):
        audit_db.clear_all()

        ev_dir = Path("data/evidence")
        if ev_dir.is_dir():
            for f in ev_dir.glob("*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass

        restore_system_nominal(audit_db)

        with st.spinner("Database wiped. Re-seeding pristine telemetry..."):
            seed_demo_simulation(60, audit_db, evidence_mgr)
        st.sidebar.success("Database wiped and re-seeded with pristine telemetry!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Metadata")
    sys_cfg = load_system_config()
    st.sidebar.info(
        f"**Camera:** {sys_cfg.camera_id}\n\n"
        f"**Machine:** {sys_cfg.machine_id}\n\n"
        f"**Vision backend:** {sys_cfg.inference.backend}\n\n"
        "**Storage:** SQLite (WAL) audit log and spool"
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Live Telemetry & Risk Stream",
        "🔍 Operator Triage Queue",
        "💥 Chaos Engineering & FMEA Diagnostics",
    ])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        latest_risk = recent_events[0]["risk_state"] if recent_events else "NOMINAL"
        latest_state = recent_events[0].get("machine_state", "RUNNING") if recent_events else "RUNNING"
        confirm_rate = operator_metrics.get("confirmation_rate", 0.0)
        confirm_rate_pct = f"{confirm_rate * 100:.1f}%" if confirm_rate is not None else "n/a (no reviews yet)"

        with col1:
            st.metric("Operational Risk", latest_risk)
        with col2:
            st.metric("Machine State", latest_state)
        with col3:
            st.metric("Pending Triage Queue", str(operator_metrics["pending_reviews"]))
        with col4:
            st.metric("Operator Confirmation Rate", confirm_rate_pct)

        st.markdown("---")

        if recent_telemetry:
            df_telem = pd.DataFrame(recent_telemetry).iloc[::-1].reset_index(drop=True)
            df_telem["Cycle"] = df_telem.index + 1

            st.subheader("1. Inspection Risk Trajectory & Anomaly Scores")
            st.line_chart(df_telem.set_index("Cycle")[["vision_score", "sensor_score"]])

            st.subheader("2. Thermal Dynamics Strip (°C)")
            st.line_chart(df_telem.set_index("Cycle")[["temperature_c"]])

            st.subheader("3. Motor Current Draw (Amperes)")
            st.line_chart(df_telem.set_index("Cycle")[["current_amps"]])

            st.subheader("4. 3-Axis Mechanical Vibration (g RMS)")
            st.line_chart(df_telem.set_index("Cycle")[["vibration_rms"]])

            st.markdown("---")
            st.subheader("Recent Ingestion Stream (Newest First)")

            display_df = df_telem.iloc[::-1].head(15)[
                [
                    "timestamp_utc",
                    "vision_score",
                    "sensor_score",
                    "vibration_rms",
                    "temperature_c",
                    "current_amps",
                ]
            ].copy()

            for col in ["vision_score", "sensor_score", "vibration_rms", "temperature_c", "current_amps"]:
                display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "0.000")

            st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.info("No continuous telemetry records ingested yet.")
            st.markdown(
                "Click below to generate realistic cyber-physical telemetry, thermal dynamics, "
                "and optical defect heatmaps:"
            )
            if st.button("🚀 Seed Demo Telemetry Data", width="stretch"):
                with st.spinner("Seeding initial live telemetry stream..."):
                    seed_demo_simulation(60, audit_db, evidence_mgr)
                st.rerun()

    with tab2:
        st.subheader("Actionable Incident Triage Queue")

        # New alerts only (aggregated decisions of an open incident do not enter the queue).
        actionable_events = audit_db.query_pending_alerts(limit=500) + [
            e for e in audit_db.query_recent_events(limit=500)
            if e.get("is_new_alert") and e.get("review_status") in ("CONFIRMED", "REJECTED")
        ]

        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            status_filter = st.selectbox(
                "Review Status Filter", ["ALL", "PENDING", "CONFIRMED", "REJECTED"], index=1
            )
        with f_col2:
            risk_filter = st.selectbox(
                "Risk Level Filter", ["ALL", "HIGH_SEVERITY", "REVIEW_REQUIRED"], index=0
            )

        filtered_events = actionable_events
        if status_filter != "ALL":
            filtered_events = [e for e in filtered_events if e.get("review_status") == status_filter]
        if risk_filter != "ALL":
            filtered_events = [e for e in filtered_events if e.get("risk_state") == risk_filter]

        if filtered_events:
            event_options = [
                f"{e['event_id'][:8]}... | {e['timestamp_utc'][11:19]} | {e['risk_state']} | {e['trigger_reason']} | Status: {e['review_status']}"
                for e in filtered_events
            ]
            selected_idx = st.selectbox(
                "Select Actionable Incident for Review",
                range(len(event_options)),
                format_func=lambda i: event_options[i],
            )
            selected_event = filtered_events[selected_idx]

            st.markdown("### Incident Investigation & Optical Evidence")
            col_img, col_meta, col_action = st.columns([2, 2, 2])

            with col_img:
                st.markdown("**Optical Defect Heatmap Overlay**")
                evidence_uri = selected_event.get("evidence_uri")
                if evidence_uri:
                    img = evidence_mgr.load_evidence(evidence_uri)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(
                            img_rgb,
                            caption=f"Evidence: {Path(evidence_uri).name}",
                            width="stretch",
                        )
                    else:
                        st.warning(f"Evidence file not found on disk at: {evidence_uri}")
                else:
                    st.info("No optical evidence image attached to this nominal event.")

            with col_meta:
                st.markdown("**Telemetry & Correlation Attributes**")
                meta_table = {
                    "Event ID": selected_event["event_id"],
                    "Timestamp (UTC)": selected_event["timestamp_utc"],
                    "Risk Classification": selected_event["risk_state"],
                    "Trigger Reason": selected_event["trigger_reason"],
                    "Vision Score (Raw / EMA)": f"{selected_event['vision_raw']:.3f} / {selected_event['vision_ema']:.3f}",
                    "Sensor Score (Raw / EMA)": f"{selected_event['sensor_raw']:.3f} / {selected_event['sensor_ema']:.3f}",
                    "Cooldown Remaining": str(selected_event["cooldown_remaining"]),
                    "Hardware Degraded": "YES" if selected_event["is_degraded"] else "NO",
                    "Correlated Frame ID": str(selected_event["frame_id"]),
                    "Correlated Sensor ID": str(selected_event["reading_id"]),
                }
                st.table(pd.DataFrame(list(meta_table.items()), columns=["Field", "Value"]))

            with col_action:
                st.markdown("**Operator Triage & Action Panel**")
                current_status = selected_event.get("review_status", "PENDING")
                st.info(f"Current Review Status: **{current_status}**")

                notes = st.text_area(
                    "Operator Remarks / Root Cause Notes",
                    value=selected_event.get("operator_notes") or "",
                    placeholder="e.g. Confirmed surface fracture on weld seam...",
                )

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Confirm Defect", width="stretch"):
                        audit_db.record_operator_review(
                            selected_event["event_id"], action="CONFIRMED", notes=notes
                        )
                        st.toast("Defect CONFIRMED and escalated to maintenance!", icon="✅")
                        st.rerun()

                with btn_col2:
                    if st.button("❌ Reject Alarm", width="stretch"):
                        audit_db.record_operator_review(
                            selected_event["event_id"], action="REJECTED", notes=notes
                        )
                        st.toast("Alert flagged as FALSE POSITIVE.", icon="❌")
                        st.rerun()
        else:
            st.info("No actionable incidents (HIGH_SEVERITY or REVIEW_REQUIRED) match the active filters.")

    with tab3:
        st.subheader("Interactive Chaos Engineering & Fault Simulation")

        st.markdown("Inject programmatic hardware and network faults into the live edge runtime loop:")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)

        with c_col1:
            if st.button("📷 Inject Camera Blur", width="stretch"):
                inject_chaos_fault_event("OPTICAL_BLUR", audit_db, evidence_mgr)
                st.toast("Injected OPTICAL_BLUR chaos fault!", icon="📷")
                st.rerun()
        with c_col2:
            if st.button("🔌 Disconnect MQTT Broker", width="stretch"):
                inject_chaos_fault_event("NETWORK_PARTITION", audit_db, evidence_mgr)
                st.toast("Injected NETWORK_PARTITION chaos fault! Disk spool active.", icon="🔌")
                st.rerun()
        with c_col3:
            if st.button("🌡️ Trigger Thermal Drift", width="stretch"):
                inject_chaos_fault_event("SENSOR_DRIFT", audit_db, evidence_mgr)
                st.toast("Injected SENSOR_DRIFT chaos fault!", icon="🌡️")
                st.rerun()
        with c_col4:
            if st.button("⚡ Current Sensor Dropout", width="stretch"):
                inject_chaos_fault_event("SENSOR_DROPOUT", audit_db, evidence_mgr)
                st.toast("Injected SENSOR_DROPOUT on current channel!", icon="⚡")
                st.rerun()

        st.markdown("")
        if st.button("🔄 Restore System to Nominal State", width="stretch"):
            restore_system_nominal(audit_db)
            st.toast("All subsystems restored to HEALTHY nominal state!", icon="🔄")
            st.rerun()

        st.markdown("---")
        st.subheader("Dynamic Subsystem FMEA Diagnostic Health Matrix")

        fmea_rows = compute_dynamic_fmea(audit_db, recent_events, recent_telemetry)
        st.dataframe(pd.DataFrame(fmea_rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
