"""Offline sensor trace replay and synthetic degradation trace generators.

The generators below produce *synthetic* traces whose shape (a healthy plateau,
an incipient drift, then a runaway phase) is loosely modeled on the qualitative
behavior described for the NASA IMS bearing and C-MAPSS turbofan run-to-failure
datasets. No values are fitted to, or sampled from, those datasets.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from loguru import logger
import numpy as np
import pandas as pd

from src.sensor_simulator import MachineState, SensorReading, composite_sensor_score


class RealSensorTraceReplay:
    """Replays historical industrial multi-modal sensor traces with calibration."""

    def __init__(
        self,
        trace_path: str | Path,
        machine_id: str = "press_unit_04",
        calibration_window_steps: int = 50,
        z_threshold: float = 3.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize trace replay loader and calibrate nominal baseline envelopes.

        Args:
            trace_path: Path to CSV or Parquet file containing telemetry traces.
            machine_id: Target machine identifier string.
            calibration_window_steps: Number of initial steps used for baseline calibration.
            z_threshold: Z-score cutoff for normalizing anomaly scores.
        """
        self.trace_path = Path(trace_path)
        self.machine_id = machine_id
        self.calib_steps = calibration_window_steps
        self.z_threshold = z_threshold
        self.weights = weights or {"vibration": 0.45, "temperature": 0.25, "current": 0.30}
        self._current_index = 0

        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_path}")

        if self.trace_path.suffix.lower() == ".parquet":
            self.df = pd.read_parquet(self.trace_path)
        else:
            self.df = pd.read_csv(self.trace_path)

        required_cols = {"vibration_rms", "temperature_c", "current_amps"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Trace data missing required columns: {missing}")

        self._calibrate()
        logger.info(
            f"Initialized RealSensorTraceReplay (rows={len(self.df)}, calib_steps={self.calib_steps})"
        )

    def _calibrate(self) -> None:
        """Derive channel means and standard deviations from calibration window."""
        if self.calib_steps < 5 or self.calib_steps > len(self.df):
            raise ValueError("calibration_window_steps must be between 5 and the trace length")
        calib_df = self.df.iloc[: self.calib_steps]
        self.means = {
            "vibration_rms": float(calib_df["vibration_rms"].mean()),
            "temperature_c": float(calib_df["temperature_c"].mean()),
            "current_amps": float(calib_df["current_amps"].mean()),
        }
        self.stds = {
            "vibration_rms": max(float(calib_df["vibration_rms"].std()), 1e-6),
            "temperature_c": max(float(calib_df["temperature_c"].std()), 1e-6),
            "current_amps": max(float(calib_df["current_amps"].std()), 1e-6),
        }

    def compute_sensor_score(self, vib: float, temp: float, curr: float) -> float:
        """Composite score using the same fusion rule as the online simulator."""
        z = {
            "vibration": (vib - self.means["vibration_rms"]) / self.stds["vibration_rms"],
            "temperature": (temp - self.means["temperature_c"]) / self.stds["temperature_c"],
            "current": (curr - self.means["current_amps"]) / self.stds["current_amps"],
        }
        score, _ = composite_sensor_score(z, self.weights, self.z_threshold)
        return score

    def step(self) -> SensorReading:
        """Return the next reading. Raises StopIteration at the end of the trace (no silent wrap-around)."""
        if self._current_index >= len(self.df):
            raise StopIteration("End of trace reached")

        row = self.df.iloc[self._current_index]
        self._current_index += 1

        vib = float(row["vibration_rms"])
        temp = float(row["temperature_c"])
        curr = float(row["current_amps"])

        # Check for NaN / dropout in trace
        missing_channels: List[str] = []
        if np.isnan(vib):
            missing_channels.append("vibration")
            vib = self.means["vibration_rms"]
        if np.isnan(temp):
            missing_channels.append("temperature")
            temp = self.means["temperature_c"]
        if np.isnan(curr):
            missing_channels.append("current")
            curr = self.means["current_amps"]

        sensor_score = self.compute_sensor_score(vib, temp, curr)
        machine_state_str = str(row.get("machine_state", "RUNNING"))
        try:
            m_state = MachineState(machine_state_str)
        except ValueError:
            m_state = MachineState.RUNNING

        step_idx = self._current_index - 1
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step_idx / 30.0)
        now_utc = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return SensorReading(
            reading_id=f"replay_{self._current_index:06d}",
            timestamp_utc=now_utc,
            machine_id=self.machine_id,
            machine_state=m_state,
            vibration_rms=round(vib, 4),
            temperature_c=round(temp, 2),
            current_amps=round(curr, 2),
            missing_channels=missing_channels,
            is_degraded=len(missing_channels) > 0,
            sensor_score=round(sensor_score, 4),
        )

    def __iter__(self) -> Iterator[SensorReading]:
        """Iterate over all rows in the trace as SensorReading objects."""
        self.reset()
        while self._current_index < len(self.df):
            yield self.step()

    def reset(self) -> None:
        """Reset replay pointer to beginning of trace."""
        self._current_index = 0

    def total_steps(self) -> int:
        """Return total number of rows in trace."""
        return len(self.df)


def generate_sample_physical_trace(
    csv_path: str | Path = "data/traces/sample_industrial_trace.csv",
    n_steps: int = 500,
    seed: int = 42,
) -> Path:
    """Generate a realistic synthetic multi-channel industrial physical trace."""
    out_file = Path(csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    time_series = []

    p1 = int(n_steps * 0.3)
    p2 = int(n_steps * 0.6)

    # 1. Warmup / Nominal Phase
    for i in range(p1):
        vib = 0.40 + rng.normal(0.0, 0.03)
        temp = 55.0 + rng.normal(0.0, 0.5)
        curr = 12.0 + rng.normal(0.0, 0.4)
        time_series.append({
            "step": i,
            "vibration_rms": max(0.01, vib),
            "temperature_c": temp,
            "current_amps": curr,
            "machine_state": "RUNNING",
        })

    # 2. Bearing Wear / Friction Anomaly Phase
    for i in range(p1, p2):
        progress = (i - p1) / max(p2 - p1, 1)
        vib = 0.40 + (0.55 * progress) + rng.normal(0.0, 0.05)
        temp = 55.0 + (18.0 * progress) + rng.normal(0.0, 0.8)
        curr = 12.0 + (6.0 * progress) + rng.normal(0.0, 0.6)
        time_series.append({
            "step": i,
            "vibration_rms": vib,
            "temperature_c": temp,
            "current_amps": curr,
            "machine_state": "RUNNING",
        })

    # 3. Intermittent Sensor Dropout & Recovery Phase
    dropout_start = p2 + int((n_steps - p2) * 0.25)
    dropout_end = p2 + int((n_steps - p2) * 0.45)
    maint_start = p2 + int((n_steps - p2) * 0.75)

    for i in range(p2, n_steps):
        is_dropout = dropout_start <= i <= dropout_end
        vib = np.nan if is_dropout else (0.42 + rng.normal(0.0, 0.04))
        temp = 58.0 + rng.normal(0.0, 0.6)
        curr = 12.5 + rng.normal(0.0, 0.4)
        m_state = "MAINTENANCE" if i >= maint_start else "RUNNING"
        time_series.append({
            "step": i,
            "vibration_rms": vib,
            "temperature_c": temp,
            "current_amps": curr,
            "machine_state": m_state,
        })

    df = pd.DataFrame(time_series)
    df.to_csv(out_file, index=False)
    logger.info(f"Generated sample physical sensor trace ({len(df)} rows) -> {out_file}")
    return out_file


def generate_ims_bearing_trace(
    csv_path: str | Path = "data/traces/ims_bearing_trace.csv",
    n_steps: int = 600,
    seed: int = 42,
) -> Path:
    """Generate a synthetic bearing run-to-failure trace (IMS-inspired shape, hand-chosen parameters).

    Vibration RMS stays near 0.35 g, drifts to about 0.80 g, then runs away to about 2.85 g,
    with secondary frictional heating. Parameters are illustrative, not fitted to the IMS data.
    """
    out_file = Path(csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    time_series = []

    p_healthy = int(n_steps * 0.40)  # Steps 0..240: Healthy operation
    p_onset = int(n_steps * 0.70)    # Steps 240..420: Incipient micro-pitting

    for i in range(n_steps):
        if i < p_healthy:
            vib_mean = 0.35 + rng.normal(0.0, 0.02)
            temp_mean = 48.0 + rng.normal(0.0, 0.4)
            curr_mean = 10.5 + rng.normal(0.0, 0.3)
        elif i < p_onset:
            progress = (i - p_healthy) / (p_onset - p_healthy)
            vib_mean = 0.35 + (0.45 * (progress ** 1.5)) + rng.normal(0.0, 0.04)
            temp_mean = 48.0 + (6.0 * progress) + rng.normal(0.0, 0.6)
            curr_mean = 10.5 + (1.5 * progress) + rng.normal(0.0, 0.4)
        else:
            progress = (i - p_onset) / (n_steps - p_onset)
            vib_mean = 0.80 + (2.05 * (progress ** 2.2)) + rng.normal(0.0, 0.08)
            temp_mean = 54.0 + (22.0 * (progress ** 1.2)) + rng.normal(0.0, 1.0)
            curr_mean = 12.0 + (4.5 * progress) + rng.normal(0.0, 0.6)

        time_series.append({
            "step": i,
            "vibration_rms": max(0.05, float(vib_mean)),
            "temperature_c": float(temp_mean),
            "current_amps": float(curr_mean),
            "machine_state": "RUNNING",
        })

    df = pd.DataFrame(time_series)
    df.to_csv(out_file, index=False)
    logger.info(f"Generated synthetic bearing trace ({len(df)} rows) -> {out_file}")
    return out_file


def generate_cmapss_turbofan_trace(
    csv_path: str | Path = "data/traces/cmapss_turbofan_trace.csv",
    n_steps: int = 600,
    seed: int = 42,
) -> Path:
    """Generate a synthetic thermal-creep degradation trace (C-MAPSS-inspired shape, hand-chosen parameters).

    Temperature rises from about 52 C to 88.5 C and current from 11.5 A to 24.8 A.
    Parameters are illustrative, not fitted to the C-MAPSS data.
    """
    out_file = Path(csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    time_series = []

    p_healthy = int(n_steps * 0.35)  # Steps 0..210: Nominal thermal baseline
    p_onset = int(n_steps * 0.65)    # Steps 210..390: Thermal drift

    for i in range(n_steps):
        if i < p_healthy:
            vib_mean = 0.42 + rng.normal(0.0, 0.03)
            temp_mean = 52.0 + rng.normal(0.0, 0.5)
            curr_mean = 11.5 + rng.normal(0.0, 0.3)
        elif i < p_onset:
            progress = (i - p_healthy) / (p_onset - p_healthy)
            vib_mean = 0.42 + (0.25 * progress) + rng.normal(0.0, 0.04)
            temp_mean = 52.0 + (14.0 * (progress ** 1.3)) + rng.normal(0.0, 0.7)
            curr_mean = 11.5 + (4.0 * progress) + rng.normal(0.0, 0.4)
        else:
            progress = (i - p_onset) / (n_steps - p_onset)
            vib_mean = 0.67 + (0.65 * (progress ** 1.8)) + rng.normal(0.0, 0.06)
            temp_mean = 66.0 + (22.5 * (progress ** 1.5)) + rng.normal(0.0, 1.2)
            curr_mean = 15.5 + (9.3 * (progress ** 1.4)) + rng.normal(0.0, 0.7)

        time_series.append({
            "step": i,
            "vibration_rms": max(0.05, float(vib_mean)),
            "temperature_c": float(temp_mean),
            "current_amps": float(curr_mean),
            "machine_state": "RUNNING",
        })

    df = pd.DataFrame(time_series)
    df.to_csv(out_file, index=False)
    logger.info(f"Generated synthetic thermal-creep trace ({len(df)} rows) -> {out_file}")
    return out_file
