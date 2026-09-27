"""Unit tests for multi-modal physical sensor simulation and anomaly scoring."""

import numpy as np
import pytest

from src.config import (
    AnomalyScoringConfig,
    AnomalyWeightsConfig,
    CurrentSensorConfig,
    MachineStateConfig,
    SensorConfig,
    SensorsConfig,
    SimulationConfig,
    TemperatureSensorConfig,
    VibrationSensorConfig,
)
from src.sensor_simulator import MachineState, SensorReading, SensorSimulator


@pytest.fixture
def sensor_config() -> SensorConfig:
    """Fixture providing a known deterministic sensor configuration."""
    return SensorConfig(
        simulation=SimulationConfig(sampling_rate_hz=30, random_seed=42),
        sensors=SensorsConfig(
            vibration=VibrationSensorConfig(baseline_rms=0.45, noise_std=0.05, fault_multiplier=3.5),
            temperature=TemperatureSensorConfig(
                ambient_celsius=25.0,
                running_target_celsius=62.0,
                thermal_inertia_k=0.02,
                drift_rate_c_per_hour=0.5,
                noise_std=0.1,
            ),
            current=CurrentSensorConfig(
                idle_amps=2.1,
                running_amps=12.8,
                fault_amps=22.4,
                noise_std=0.2,
            ),
        ),
        machine_states={
            "IDLE": MachineStateConfig(load_factor=0.1),
            "RUNNING": MachineStateConfig(load_factor=1.0),
            "MAINTENANCE": MachineStateConfig(load_factor=0.0),
            "FAULT": MachineStateConfig(load_factor=1.8),
        },
        anomaly_scoring=AnomalyScoringConfig(
            zscore_threshold=3.0,
            weights=AnomalyWeightsConfig(vibration=0.45, temperature=0.25, current=0.30),
        ),
    )


@pytest.fixture
def simulator(sensor_config: SensorConfig) -> SensorSimulator:
    """Fixture providing an instantiated SensorSimulator."""
    return SensorSimulator(config=sensor_config, machine_id="press_unit_04", seed=42)


def test_thermal_physics_bounded_convergence(simulator: SensorSimulator) -> None:
    """Test that underlying physical temperature converges asymptotically to target without unbounded random walk."""
    simulator.reset()
    target_temp = simulator.config.sensors.temperature.running_target_celsius

    # Simulate 300 steps in RUNNING state
    for _ in range(300):
        simulator.step(machine_state=MachineState.RUNNING)

    # Physical internal state must be strictly bounded close to equilibrium target
    assert abs(simulator.internal_temp_c - target_temp) < 2.0


def test_injected_fault_triggers_high_composite_score(simulator: SensorSimulator) -> None:
    """Test that inject_fault=True causes composite sensor_score to exceed 0.70."""
    simulator.reset()

    for _ in range(50):
        simulator.step(machine_state=MachineState.RUNNING)

    fault_reading = simulator.step(machine_state=MachineState.RUNNING, inject_fault=True)

    assert isinstance(fault_reading, SensorReading)
    assert fault_reading.sensor_score > 0.70
    assert fault_reading.sensor_score <= 1.0
    assert fault_reading.vibration_rms > simulator.config.sensors.vibration.baseline_rms * 2.0
    assert fault_reading.current_amps > simulator.config.sensors.current.running_amps * 1.3
    assert fault_reading.sensor_breakdown["vibration_zscore"] > 3.0
    assert fault_reading.sensor_breakdown["current_zscore"] > 3.0


def test_state_transitions_load_and_currents(simulator: SensorSimulator) -> None:
    """Test physical signals in IDLE and MAINTENANCE states."""
    simulator.reset()

    # Idle step
    idle_reading = simulator.step(machine_state=MachineState.IDLE)
    assert idle_reading.vibration_rms < simulator.config.sensors.vibration.baseline_rms * 0.5
    assert abs(idle_reading.current_amps - simulator.config.sensors.current.idle_amps) < 1.0
    assert idle_reading.is_degraded is False

    # Maintenance step
    maint_reading = simulator.step(machine_state=MachineState.MAINTENANCE)
    assert maint_reading.vibration_rms < 0.2
    assert maint_reading.current_amps < 1.0


def test_selective_channel_dropout(simulator: SensorSimulator) -> None:
    """Test that selective dropout (e.g. only current) imputes dropped channel while keeping other channels intact."""
    simulator.reset()

    # Warm up with a normal step to record valid readings
    normal_reading = simulator.step(machine_state=MachineState.RUNNING)
    last_valid_cur = normal_reading.current_amps
    assert normal_reading.is_degraded is False
    assert normal_reading.missing_channels == []

    # Inject dropout specifically on 'current' channel
    dropout_reading = simulator.step(machine_state=MachineState.RUNNING, simulate_dropout=["current"])

    assert dropout_reading.is_degraded is True
    assert dropout_reading.missing_channels == ["current"]
    # Current should be imputed to the last valid measurement
    assert dropout_reading.current_amps == last_valid_cur
    # Vibration and temperature should be active and valid
    assert dropout_reading.vibration_rms > 0.0
    assert dropout_reading.temperature_c > 0.0
    assert not np.isnan(dropout_reading.sensor_score)


def test_multi_channel_dropout_initial(simulator: SensorSimulator) -> None:
    """Test dropout on multiple channels before any step has run falls back to baseline."""
    simulator.reset()

    dropout_reading = simulator.step(
        machine_state=MachineState.RUNNING,
        simulate_dropout=["vibration", "temperature", "current"],
    )

    assert dropout_reading.is_degraded is True
    assert set(dropout_reading.missing_channels) == {"vibration", "temperature", "current"}
    # Before any valid sample exists, held values fall back to the RUNNING baselines,
    # so a dropout cannot by itself look like an anomaly.
    cfg = simulator.config
    assert dropout_reading.vibration_rms == cfg.sensors.vibration.baseline_rms * cfg.machine_states["RUNNING"].load_factor
    assert dropout_reading.temperature_c == cfg.sensors.temperature.running_target_celsius
    assert dropout_reading.current_amps == cfg.sensors.current.running_amps
    assert 0.0 <= dropout_reading.sensor_score <= 1.0


def test_deterministic_output_with_fixed_seed(sensor_config: SensorConfig) -> None:
    """Test reproducibility across multiple simulator instances with fixed random seed."""
    sim1 = SensorSimulator(config=sensor_config, seed=777)
    sim2 = SensorSimulator(config=sensor_config, seed=777)

    readings1 = [sim1.step(machine_state=MachineState.RUNNING) for _ in range(30)]
    readings2 = [sim2.step(machine_state=MachineState.RUNNING) for _ in range(30)]

    for r1, r2 in zip(readings1, readings2):
        assert pytest.approx(r1.vibration_rms, rel=1e-5) == r2.vibration_rms
        assert pytest.approx(r1.temperature_c, rel=1e-5) == r2.temperature_c
        assert pytest.approx(r1.current_amps, rel=1e-5) == r2.current_amps
        assert pytest.approx(r1.sensor_score, rel=1e-5) == r2.sensor_score


def test_reset_behavior(simulator: SensorSimulator) -> None:
    """Test that simulator reset restores initial thermal and counters."""
    simulator.reset()
    for _ in range(50):
        simulator.step(machine_state=MachineState.FAULT)

    assert simulator.internal_temp_c > 30.0
    assert simulator.step_count == 50

    simulator.reset()
    assert simulator.internal_temp_c == simulator.config.sensors.temperature.ambient_celsius
    assert simulator.step_count == 0
    assert simulator.elapsed_time_hours == 0.0
    assert simulator._last_valid_channels == {}


def test_sensor_reading_data_contract(simulator: SensorSimulator) -> None:
    """Test that SensorReading complies with all required UUID, ISO timestamp, and breakdown fields."""
    simulator.reset()
    reading = simulator.step(machine_state=MachineState.RUNNING)

    assert len(reading.reading_id) == 36  # UUID string format
    assert "T" in reading.timestamp_utc and reading.timestamp_utc.endswith("Z")
    assert reading.machine_id == "press_unit_04"
    assert "vibration_zscore" in reading.sensor_breakdown
    assert "temperature_zscore" in reading.sensor_breakdown
    assert "current_zscore" in reading.sensor_breakdown