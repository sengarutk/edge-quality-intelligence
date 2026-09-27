"""Multi-Modal Industrial Physical Sensor Simulator.

Simulates physics-driven vibration, thermal inertia, and electrical current dynamics
for edge machine state health monitoring and composite anomaly scoring.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config import SensorConfig, load_sensor_config


SENSOR_CHANNELS = ("vibration", "temperature", "current")


def composite_sensor_score(
    zscores: Dict[str, float], weights: Dict[str, float], z_threshold: float
) -> Tuple[float, float]:
    """Fuse per-channel z-scores into one score in [0, 1].

    Only the excess above ``z_threshold`` counts (one-sided), scaled so that three
    extra standard deviations add one unit. The weighted sum ``s_raw`` is mapped
    to ``1 - exp(-s_raw)``. Used by both the simulator and trace replay so the
    policy threshold has the same meaning for both sources.
    """
    s_raw = 0.0
    for ch, z in zscores.items():
        s_raw += weights.get(ch, 0.0) * max(0.0, (z - z_threshold) / 3.0)
    return float(np.clip(1.0 - np.exp(-s_raw), 0.0, 1.0)), float(s_raw)


class MachineState(str, Enum):
    """Operational machine states in the industrial inspection cell."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    MAINTENANCE = "MAINTENANCE"
    FAULT = "FAULT"


class SensorReading(BaseModel):
    """Multi-modal sensor telemetry packet at a discrete simulation step."""
    model_config = ConfigDict(extra="forbid")

    reading_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Globally unique reading identifier (UUIDv4)."
    )
    timestamp_utc: str = Field(..., description="ISO-8601 UTC formatted timestamp string (YYYY-MM-DDTHH:MM:SS.fffZ).")
    machine_id: str = Field(..., description="Machine unit identifier.")
    machine_state: MachineState = Field(..., description="Operational state of machine during sampling.")
    vibration_rms: float = Field(..., ge=0.0, description="Measured vibration root mean square amplitude (g).")
    temperature_c: float = Field(..., description="Measured internal temperature in Celsius.")
    current_amps: float = Field(..., ge=0.0, description="Measured electrical current in Amperes.")
    missing_channels: List[str] = Field(
        default_factory=list, description="List of channel names with dropout or communication loss."
    )
    is_degraded: bool = Field(..., description="True if any sensor channel is missing or degraded.")
    sensor_score: float = Field(..., ge=0.0, le=1.0, description="Normalized composite anomaly score in [0.0, 1.0].")
    sensor_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Standardized Z-scores and raw measurements per sensor channel."
    )


class SensorSimulator:
    """Physics-inspired first-order sensor simulator for edge machine condition monitoring."""

    def __init__(
        self,
        config: Optional[SensorConfig] = None,
        machine_id: str = "press_unit_04",
        seed: Optional[int] = None,
        sim_epoch: Optional[datetime] = None,
    ) -> None:
        """Initialize the sensor simulator with physics and telemetry configuration.

        Args:
            config: Optional SensorConfig instance. If None, default config is loaded from disk.
            machine_id: Unique identifier for the monitored physical machine.
            seed: Optional explicit random seed. If None, uses config.simulation.random_seed.
            sim_epoch: If given, timestamps follow simulated time (epoch + step / rate) instead of the wall clock,
                which makes generated traces reproducible.
        """
        self.sim_epoch = sim_epoch
        self.config = config or load_sensor_config()
        self.machine_id = machine_id
        self._seed = seed if seed is not None else self.config.simulation.random_seed
        self._rng = np.random.RandomState(self._seed)

        # Pure physical state variables (free of measurement noise)
        self._internal_temp_c: float = self.config.sensors.temperature.ambient_celsius
        self.elapsed_time_hours: float = 0.0
        self.step_count: int = 0
        self._last_valid_channels: Dict[str, float] = {}

        logger.info(
            f"Initialized SensorSimulator (machine_id={self.machine_id}, "
            f"rate={self.config.simulation.sampling_rate_hz}Hz, seed={self._seed})"
        )

    @property
    def internal_temp_c(self) -> float:
        """Get the current underlying physical temperature in Celsius."""
        return self._internal_temp_c

    def reset(self) -> None:
        """Reset internal temperature, drift, step count, and random seed state."""
        self._internal_temp_c = self.config.sensors.temperature.ambient_celsius
        self.elapsed_time_hours = 0.0
        self.step_count = 0
        self._last_valid_channels = {}
        self._rng = np.random.RandomState(self._seed)
        logger.info("SensorSimulator state successfully reset to initial ambient conditions.")

    def _get_target_temperature(self, machine_state: MachineState, inject_fault: bool) -> float:
        """Calculate equilibrium target temperature for the current state."""
        ambient = self.config.sensors.temperature.ambient_celsius
        running_target = self.config.sensors.temperature.running_target_celsius

        if machine_state == MachineState.FAULT or inject_fault:
            return running_target + 25.0
        elif machine_state == MachineState.RUNNING:
            return running_target
        elif machine_state == MachineState.IDLE:
            return ambient + 5.0
        elif machine_state == MachineState.MAINTENANCE:
            return ambient
        return ambient

    def _calculate_vibration(self, machine_state: MachineState, inject_fault: bool) -> float:
        """Compute synthetic vibration RMS based on load factor and mechanical faults."""
        state_key = machine_state.value
        load_factor = self.config.machine_states.get(state_key, self.config.machine_states["IDLE"]).load_factor
        baseline_rms = self.config.sensors.vibration.baseline_rms
        noise_std = self.config.sensors.vibration.noise_std

        if machine_state == MachineState.FAULT or inject_fault:
            fault_boost = self.config.sensors.vibration.fault_multiplier - 1.0
        else:
            fault_boost = 0.0

        vibration = baseline_rms * load_factor * (1.0 + fault_boost) + self._rng.normal(0.0, noise_std)
        return float(max(0.0, vibration))

    def _calculate_current(self, machine_state: MachineState, inject_fault: bool) -> float:
        """Compute synthetic electrical current draw in Amperes."""
        noise_std = self.config.sensors.current.noise_std

        if machine_state == MachineState.FAULT or inject_fault:
            base_amps = self.config.sensors.current.fault_amps
        elif machine_state == MachineState.RUNNING:
            base_amps = self.config.sensors.current.running_amps
        elif machine_state == MachineState.IDLE:
            base_amps = self.config.sensors.current.idle_amps
        elif machine_state == MachineState.MAINTENANCE:
            base_amps = 0.0
        else:
            base_amps = self.config.sensors.current.idle_amps

        current = base_amps + self._rng.normal(0.0, noise_std)
        return float(max(0.0, current))

    def _update_physics_temperature(self, machine_state: MachineState, inject_fault: bool, delta_t_hours: float) -> float:
        """Update pure physical temperature without adding measurement noise to the state."""
        target_temp = self._get_target_temperature(machine_state, inject_fault)
        k = self.config.sensors.temperature.thermal_inertia_k
        drift_rate = self.config.sensors.temperature.drift_rate_c_per_hour

        # Physical state transition: \bar{T}_t = \bar{T}_{t-1} + k * (T_target - \bar{T}_{t-1}) + drift * dt
        self._internal_temp_c = (
            self._internal_temp_c
            + k * (target_temp - self._internal_temp_c)
            + (drift_rate * delta_t_hours)
        )
        return self._internal_temp_c

    def _observe_temperature(self) -> float:
        """Generate noisy sensor measurement from pure physical thermal state."""
        noise_std = self.config.sensors.temperature.noise_std
        noisy_temp = self._internal_temp_c + self._rng.normal(0.0, noise_std)
        return float(noisy_temp)

    def _compute_zscores_and_score(
        self, vibration: float, temperature: float, current: float
    ) -> Tuple[Dict[str, float], float]:
        """Compute standardized Z-scores and fused composite anomaly score."""
        # Operational baseline references
        baseline_vib_mean = self.config.sensors.vibration.baseline_rms * self.config.machine_states["RUNNING"].load_factor
        baseline_vib_std = max(1e-6, self.config.sensors.vibration.noise_std)

        baseline_temp_mean = self.config.sensors.temperature.running_target_celsius
        baseline_temp_std = max(1e-6, self.config.sensors.temperature.noise_std)

        baseline_cur_mean = self.config.sensors.current.running_amps
        baseline_cur_std = max(1e-6, self.config.sensors.current.noise_std)

        # Standardized Z-scores
        z_vib = (vibration - baseline_vib_mean) / baseline_vib_std
        z_temp = (temperature - baseline_temp_mean) / baseline_temp_std
        z_cur = (current - baseline_cur_mean) / baseline_cur_std

        weights = self.config.anomaly_scoring.weights
        sensor_score, s_raw = composite_sensor_score(
            {"vibration": z_vib, "temperature": z_temp, "current": z_cur},
            {"vibration": weights.vibration, "temperature": weights.temperature, "current": weights.current},
            self.config.anomaly_scoring.zscore_threshold,
        )

        breakdown = {
            "vibration_rms": float(vibration),
            "vibration_zscore": float(z_vib),
            "temperature_c": float(temperature),
            "temperature_zscore": float(z_temp),
            "current_amps": float(current),
            "current_zscore": float(z_cur),
            "raw_excess_sum": float(s_raw),
        }

        return breakdown, sensor_score

    def step(
        self,
        machine_state: MachineState = MachineState.RUNNING,
        inject_fault: bool = False,
        simulate_dropout: Optional[List[str]] = None,
        temperature_offset_c: float = 0.0,
    ) -> SensorReading:
        """Advance the multi-modal simulation by one discrete time step.

        Args:
            machine_state: Target machine state for this step.
            inject_fault: If True, injects mechanical and electrical fault conditions.
            simulate_dropout: Optional list of channel names to simulate dropout for
                              (subset of SENSOR_CHANNELS, e.g. ["current"]).
            temperature_offset_c: Additive measurement drift on the temperature channel.

        Returns:
            SensorReading telemetry packet conforming to updated data contract.
        """
        sampling_rate = self.config.simulation.sampling_rate_hz
        delta_t_s = 1.0 / sampling_rate
        delta_t_hours = delta_t_s / 3600.0

        self.elapsed_time_hours += delta_t_hours
        self.step_count += 1
        if self.sim_epoch is not None:
            now = self.sim_epoch + timedelta(seconds=self.step_count * delta_t_s)
        else:
            now = datetime.now(timezone.utc)
        now_utc = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Update physical thermal state
        self._update_physics_temperature(machine_state, inject_fault, delta_t_hours)

        # Generate physical measurements
        true_vib = self._calculate_vibration(machine_state, inject_fault)
        true_cur = self._calculate_current(machine_state, inject_fault)
        true_temp = self._observe_temperature() + float(temperature_offset_c)

        # Handle selective per-channel dropouts
        missing_channels = sorted(set(simulate_dropout)) if simulate_dropout else []
        unknown = set(missing_channels) - set(SENSOR_CHANNELS)
        if unknown:
            raise ValueError(f"Unknown sensor channel(s) {sorted(unknown)}; expected a subset of {SENSOR_CHANNELS}")
        is_degraded = len(missing_channels) > 0

        # Vibration channel
        if "vibration" in missing_channels:
            eff_vib = self._last_valid_channels.get(
                "vibration",
                self.config.sensors.vibration.baseline_rms * self.config.machine_states["RUNNING"].load_factor,
            )
        else:
            eff_vib = true_vib
            self._last_valid_channels["vibration"] = true_vib

        # Temperature channel
        if "temperature" in missing_channels:
            eff_temp = self._last_valid_channels.get(
                "temperature", self.config.sensors.temperature.running_target_celsius
            )
        else:
            eff_temp = true_temp
            self._last_valid_channels["temperature"] = true_temp

        # Current channel
        if "current" in missing_channels:
            eff_cur = self._last_valid_channels.get(
                "current", self.config.sensors.current.running_amps
            )
        else:
            eff_cur = true_cur
            self._last_valid_channels["current"] = true_cur

        breakdown, sensor_score = self._compute_zscores_and_score(eff_vib, eff_temp, eff_cur)

        reading = SensorReading(
            reading_id=str(uuid.uuid4()),
            timestamp_utc=now_utc,
            machine_id=self.machine_id,
            machine_state=machine_state,
            vibration_rms=eff_vib,
            temperature_c=eff_temp,
            current_amps=eff_cur,
            missing_channels=missing_channels,
            is_degraded=is_degraded,
            sensor_score=sensor_score,
            sensor_breakdown=breakdown,
        )
        return reading