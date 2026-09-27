"""Unit tests for configuration loaders and Pydantic validation."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import (
    ConfirmationWindowConfig,
    CooldownConfig,
    MachineStateGatingConfig,
    MQTTConfig,
    PolicyConfig,
    SensorConfig,
    SystemConfig,
    TemporalSmoothingConfig,
    ThresholdsConfig,
    load_config,
    load_mqtt_config,
    load_policy_config,
    load_sensor_config,
    load_system_config,
    load_yaml,
)


def test_load_sensor_config() -> None:
    """Test loading and validating sensor_config.yaml."""
    cfg = load_sensor_config("configs/sensor_config.yaml")
    assert isinstance(cfg, SensorConfig)
    assert cfg.simulation.sampling_rate_hz == 30
    assert cfg.sensors.vibration.baseline_rms == 0.45
    assert cfg.sensors.temperature.running_target_celsius == 62.0
    assert cfg.anomaly_scoring.zscore_threshold == 3.0
    assert "RUNNING" in cfg.machine_states


def test_load_system_config() -> None:
    """Test loading and validating system_config.yaml."""
    cfg = load_system_config("configs/system_config.yaml")
    assert isinstance(cfg, SystemConfig)
    assert cfg.camera_id == "line1_overhead_cam01"
    assert cfg.machine_id == "press_unit_04"
    assert cfg.optical_health.blur_laplacian_threshold == 100.0
    assert cfg.inference.backend == "mock"
    assert cfg.inference.input_resolution == [224, 224]


def test_load_policy_config() -> None:
    """Test loading and validating policy_config.yaml."""
    cfg = load_policy_config("configs/policy_config.yaml")
    assert isinstance(cfg, PolicyConfig)

    assert isinstance(cfg.temporal_smoothing, TemporalSmoothingConfig)
    assert cfg.temporal_smoothing.alpha_vision == 0.35
    assert cfg.temporal_smoothing.alpha_sensor == 0.25

    assert isinstance(cfg.confirmation_window, ConfirmationWindowConfig)
    assert cfg.confirmation_window.window_size_n == 10
    assert cfg.confirmation_window.consecutive_k == 4

    assert isinstance(cfg.cooldown, CooldownConfig)
    assert cfg.cooldown.cooldown_steps == 15

    assert isinstance(cfg.thresholds, ThresholdsConfig)
    assert cfg.thresholds.vision_medium == 0.50
    assert cfg.thresholds.vision_high == 0.80
    assert cfg.thresholds.sensor_anomaly == 0.70
    assert cfg.thresholds.cross_modal_divergence == 0.45

    assert isinstance(cfg.machine_state_gating, MachineStateGatingConfig)
    assert cfg.machine_state_gating.suppress_high_severity_on_idle is True
    assert cfg.machine_state_gating.suppress_high_severity_on_maintenance is True


def test_load_mqtt_config() -> None:
    """Test loading and validating mqtt_config.yaml."""
    cfg = load_mqtt_config("configs/mqtt_config.yaml")
    assert isinstance(cfg, MQTTConfig)
    assert cfg.broker.port == 1883
    assert cfg.topics.risk_events == "inspection/line1/risk"
    assert cfg.qos.risk_events == 1
    assert cfg.spooler.max_spool_records == 50000


def test_invalid_policy_validation_errors() -> None:
    """Test validation errors for out-of-bound policy configuration parameters."""
    with pytest.raises(ValidationError):
        TemporalSmoothingConfig(alpha_vision=-0.1)

    with pytest.raises(ValidationError):
        TemporalSmoothingConfig(alpha_sensor=1.5)

    with pytest.raises(ValidationError):
        ConfirmationWindowConfig(window_size_n=0)

    with pytest.raises(ValidationError):
        ConfirmationWindowConfig(consecutive_k=-1)

    with pytest.raises(ValidationError):
        ThresholdsConfig(vision_high=1.2)

    with pytest.raises(ValidationError):
        ThresholdsConfig(sensor_anomaly=-0.5)


def test_load_nonexistent_file() -> None:
    """Test error handling when loading nonexistent config file."""
    with pytest.raises(FileNotFoundError):
        load_yaml("configs/nonexistent_config.yaml")

    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent_config.yaml", SystemConfig)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    """Test error handling when YAML file has syntax errors."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("simulation:\n  sampling_rate_hz: [unclosed list", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_yaml(bad_yaml)


def test_validation_error_on_invalid_sensor_schema(tmp_path: Path) -> None:
    """Test that schema violations raise Pydantic ValidationError."""
    invalid_sensor = tmp_path / "invalid_sensor.yaml"
    invalid_sensor.write_text(
        """
simulation:
  sampling_rate_hz: -5
  random_seed: 42
sensors:
  vibration:
    baseline_rms: -0.1
    noise_std: 0.05
    fault_multiplier: 0.5
  temperature:
    ambient_celsius: 25.0
    running_target_celsius: 62.0
    thermal_inertia_k: 2.5
    drift_rate_c_per_hour: 0.5
    noise_std: 0.3
  current:
    idle_amps: -1.0
    running_amps: 12.8
    fault_amps: 22.4
    noise_std: 0.4
machine_states:
  IDLE:
    load_factor: 0.1
anomaly_scoring:
  zscore_threshold: 3.0
  weights:
    vibration: 0.45
    temperature: 0.25
    current: 0.30
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(invalid_sensor, SensorConfig)