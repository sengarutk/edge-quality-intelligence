"""Configuration subsystem for Edge Inspection Runtime.

Provides strict Pydantic V2 models and cached configuration loaders.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml
from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "mvtec_ad"
CONFIG_DIR = ROOT_DIR / "configs"


class PolicyMode(str, Enum):
    """Configurable decision policy variants for experimental ablation analysis."""
    BASELINE = "BASELINE"
    EMA_ONLY = "EMA_ONLY"
    EMA_KOFN = "EMA_KOFN"
    NO_COOLDOWN = "NO_COOLDOWN"
    NO_FUSION = "NO_FUSION"
    NO_DIVERGENCE = "NO_DIVERGENCE"
    NO_STATE_GATING = "NO_STATE_GATING"
    FULL_POLICY = "FULL_POLICY"
    # External alarm-management baselines (src/runtime/alarm_baselines.py), not policy variants.
    DELAY_TIMER = "DELAY_TIMER"
    EMA_HYSTERESIS = "EMA_HYSTERESIS"
    DECISION_FUSION = "DECISION_FUSION"


class SimulationConfig(BaseModel):
    """Configuration for sensor simulation execution."""
    model_config = ConfigDict(extra="forbid")

    sampling_rate_hz: int = Field(default=30, gt=0, description="Simulation sampling rate in Hertz.")
    random_seed: int = Field(default=42, description="Random seed for reproducible physical noise generation.")


class VibrationSensorConfig(BaseModel):
    """Configuration for vibration sensor physics."""
    model_config = ConfigDict(extra="forbid")

    baseline_rms: float = Field(..., gt=0.0, description="Nominal RMS vibration amplitude (g).")
    noise_std: float = Field(..., ge=0.0, description="Gaussian noise standard deviation.")
    fault_multiplier: float = Field(..., gt=1.0, description="Amplitude multiplier during mechanical fault conditions.")


class TemperatureSensorConfig(BaseModel):
    """Configuration for thermal inertia and environment."""
    model_config = ConfigDict(extra="forbid")

    ambient_celsius: float = Field(default=25.0, description="Ambient baseline temperature in Celsius.")
    running_target_celsius: float = Field(default=62.0, description="Equilibrium operational temperature in Celsius.")
    thermal_inertia_k: float = Field(..., gt=0.0, le=1.0, description="Newtonian cooling/heating coefficient per step.")
    drift_rate_c_per_hour: float = Field(default=0.5, ge=0.0, description="Long-term thermal drift in deg C per hour.")
    noise_std: float = Field(..., ge=0.0, description="Thermal sensor measurement noise standard deviation.")


class CurrentSensorConfig(BaseModel):
    """Configuration for electrical current consumption."""
    model_config = ConfigDict(extra="forbid")

    idle_amps: float = Field(..., ge=0.0, description="Baseline current draw when machine is idle (A).")
    running_amps: float = Field(..., gt=0.0, description="Nominal operational current draw (A).")
    fault_amps: float = Field(..., gt=0.0, description="Electrical current draw during overload/fault condition (A).")
    noise_std: float = Field(..., ge=0.0, description="Current measurement noise standard deviation.")


class SensorsConfig(BaseModel):
    """Container for individual sensor configurations."""
    model_config = ConfigDict(extra="forbid")

    vibration: VibrationSensorConfig
    temperature: TemperatureSensorConfig
    current: CurrentSensorConfig


class MachineStateConfig(BaseModel):
    """Load factor configuration per operational machine state."""
    model_config = ConfigDict(extra="forbid")

    load_factor: float = Field(..., ge=0.0, description="Normalized mechanical load factor.")


class AnomalyWeightsConfig(BaseModel):
    """Sensor weighting factors for composite anomaly score calculation."""
    model_config = ConfigDict(extra="forbid")

    vibration: float = Field(..., ge=0.0, le=1.0, description="Weight assigned to vibration Z-score.")
    temperature: float = Field(..., ge=0.0, le=1.0, description="Weight assigned to temperature Z-score.")
    current: float = Field(..., ge=0.0, le=1.0, description="Weight assigned to current Z-score.")


class AnomalyScoringConfig(BaseModel):
    """Anomaly scoring thresholds and fusion weights."""
    model_config = ConfigDict(extra="forbid")

    zscore_threshold: float = Field(default=3.0, ge=0.0, description="Z-score threshold for abnormality detection.")
    weights: AnomalyWeightsConfig


class SensorConfig(BaseModel):
    """Top-level configuration schema for sensor simulation and scoring."""
    model_config = ConfigDict(extra="forbid")

    simulation: SimulationConfig
    sensors: SensorsConfig
    machine_states: Dict[str, MachineStateConfig]
    anomaly_scoring: AnomalyScoringConfig


class OpticalHealthConfig(BaseModel):
    """Optical health thresholds for visual inspection stream validation."""
    model_config = ConfigDict(extra="forbid")

    blur_laplacian_threshold: float = Field(
        default=100.0, ge=0.0, description="Variance of Laplacian threshold below which frame is flagged blurred."
    )
    dark_frame_mean_threshold: float = Field(
        default=15.0, ge=0.0, description="Mean intensity threshold below which frame is flagged occluded dark."
    )
    bright_frame_mean_threshold: float = Field(
        default=245.0, le=255.0, description="Mean intensity threshold above which frame is flagged occluded bright."
    )


class InferenceConfig(BaseModel):
    """Inference execution settings for vision inspection.

    ``backend`` selects the scorer: ``mock`` (synthetic scores for unit tests and demos),
    ``patchcore`` (a fitted PatchCore memory bank saved with PatchCore.save) or ``onnx``.
    Raw distances are mapped to [0, 1] with ``score = clip(0.5 * d / score_reference, 0, 1)``,
    where ``score_reference`` is the 99th percentile of held-out nominal distances, so that
    the nominal 99th percentile lands on 0.5.
    """
    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="mock", pattern="^(mock|patchcore|onnx)$", description="Scoring backend.")
    mock_mode: Optional[bool] = Field(default=None, description="Deprecated; use backend.")
    mock_latency_ms: float = Field(default=0.0, ge=0.0, description="Artificial delay of the mock backend (ms).")
    input_resolution: List[int] = Field(
        default_factory=lambda: [224, 224], description="Image resolution [H, W] expected by the vision model."
    )
    model_path: Optional[str] = Field(default=None, description="Model artifact (PatchCore .pt or ONNX file).")
    score_reference: Optional[float] = Field(
        default=None, gt=0.0, description="Held-out nominal 99th-percentile distance used for normalization."
    )
    device: Optional[str] = Field(default=None, description="Torch device for the patchcore backend.")

    @model_validator(mode="after")
    def _legacy_mock_flag(self) -> "InferenceConfig":
        if self.mock_mode is False and self.backend == "mock":
            self.backend = "onnx"
        if self.backend != "mock" and (self.model_path is None or self.score_reference is None):
            raise ValueError("patchcore/onnx backends require model_path and score_reference")
        return self


class SystemConfig(BaseModel):
    """System-level runtime configuration."""
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(default="line1_overhead_cam01", description="Identifier for visual inspection camera.")
    machine_id: str = Field(default="press_unit_04", description="Identifier for machine unit monitored.")
    optical_health: OpticalHealthConfig = Field(default_factory=OpticalHealthConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


class TemporalSmoothingConfig(BaseModel):
    """Exponential moving average temporal smoothing configuration."""
    model_config = ConfigDict(extra="forbid")

    alpha_vision: float = Field(default=0.35, gt=0.0, le=1.0, description="EMA smoothing factor for vision scores.")
    alpha_sensor: float = Field(default=0.25, gt=0.0, le=1.0, description="EMA smoothing factor for sensor scores.")


class ConfirmationWindowConfig(BaseModel):
    """k-of-N confirmation window: an anomaly is confirmed once at least k of the
    last N smoothed samples exceed the threshold (the k samples need not be consecutive)."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    window_size_n: int = Field(default=10, ge=1, description="Sliding window length N.")
    required_k: int = Field(
        default=4,
        ge=1,
        validation_alias=AliasChoices("required_k", "consecutive_k"),
        description="Minimum number of exceedances k within the last N samples.",
    )

    @property
    def consecutive_k(self) -> int:
        """Backward-compatible alias for ``required_k`` (the samples are not required to be consecutive)."""
        return self.required_k

    @model_validator(mode="after")
    def _check_k_le_n(self) -> "ConfirmationWindowConfig":
        if self.required_k > self.window_size_n:
            raise ValueError(f"required_k ({self.required_k}) must not exceed window_size_n ({self.window_size_n})")
        return self


class CooldownConfig(BaseModel):
    """Alert suppression cooldown settings."""
    model_config = ConfigDict(extra="forbid")

    cooldown_steps: int = Field(default=15, ge=0, description="Cooldown suppression steps following an alert.")


class ThresholdsConfig(BaseModel):
    """Operational anomaly detection thresholds."""
    model_config = ConfigDict(extra="forbid")

    vision_medium: float = Field(default=0.50, ge=0.0, le=1.0, description="Medium severity visual threshold.")
    vision_high: float = Field(default=0.80, ge=0.0, le=1.0, description="High severity visual threshold.")
    sensor_anomaly: float = Field(default=0.70, ge=0.0, le=1.0, description="Composite sensor anomaly threshold.")
    cross_modal_divergence: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Max divergence allowed between vision and physical telemetry."
    )

    @model_validator(mode="after")
    def _check_ordering(self) -> "ThresholdsConfig":
        if self.vision_medium >= self.vision_high:
            raise ValueError("vision_medium must be strictly lower than vision_high")
        return self


class MachineStateGatingConfig(BaseModel):
    """Gating rules for suppressing alerts during non-operational states."""
    model_config = ConfigDict(extra="forbid")

    suppress_high_severity_on_idle: bool = Field(
        default=True, description="Suppress high severity alerts during IDLE state."
    )
    suppress_high_severity_on_maintenance: bool = Field(
        default=True, description="Suppress high severity alerts during MAINTENANCE state."
    )


class PolicyConfig(BaseModel):
    """Top-level policy configuration schema."""
    model_config = ConfigDict(extra="forbid")

    policy_mode: PolicyMode = Field(
        default=PolicyMode.FULL_POLICY, description="Active decision policy variant."
    )
    temporal_smoothing: TemporalSmoothingConfig = Field(default_factory=TemporalSmoothingConfig)
    confirmation_window: ConfirmationWindowConfig = Field(default_factory=ConfirmationWindowConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    machine_state_gating: MachineStateGatingConfig = Field(default_factory=MachineStateGatingConfig)


class MQTTBrokerConfig(BaseModel):
    """MQTT broker connection parameters."""
    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost", description="MQTT broker hostname or IP.")
    port: int = Field(default=1883, gt=0, le=65535, description="MQTT broker TCP port.")
    keepalive: int = Field(default=60, gt=0, description="Keepalive timeout in seconds.")
    reconnect_delay_min_s: float = Field(default=1.0, ge=0.1, description="Minimum exponential backoff reconnect delay.")
    reconnect_delay_max_s: float = Field(default=30.0, ge=1.0, description="Maximum exponential backoff reconnect delay.")
    client_id_prefix: str = Field(default="edge_inspector", description="Client identifier prefix.")
    client_id: Optional[str] = Field(
        default=None,
        description="Fixed client identifier. A stable id lets the broker keep the session across reconnects.",
    )
    clean_session: bool = Field(
        default=False, description="If False the broker retains QoS>=1 session state across reconnects."
    )

    @model_validator(mode="after")
    def _check_backoff(self) -> "MQTTBrokerConfig":
        if self.reconnect_delay_min_s > self.reconnect_delay_max_s:
            raise ValueError("reconnect_delay_min_s must not exceed reconnect_delay_max_s")
        return self


class MQTTTopicsConfig(BaseModel):
    """Topic hierarchy definitions for edge inspection events."""
    model_config = ConfigDict(extra="forbid")

    risk_events: str = Field(default="inspection/line1/risk", description="Topic for evaluated risk events.")
    telemetry: str = Field(default="inspection/line1/telemetry", description="Topic for raw sensor/vision telemetry.")
    health: str = Field(default="inspection/line1/health", description="Topic for system component health statuses.")
    heartbeat: str = Field(default="inspection/line1/heartbeat", description="Topic for periodic liveness heartbeats.")


class MQTTQoSConfig(BaseModel):
    """Quality of Service levels for publication topics."""
    model_config = ConfigDict(extra="forbid")

    risk_events: int = Field(default=1, ge=0, le=2, description="QoS for risk decisions.")
    telemetry: int = Field(default=0, ge=0, le=2, description="QoS for continuous telemetry.")
    health: int = Field(default=1, ge=0, le=2, description="QoS for health status updates.")
    heartbeat: int = Field(default=0, ge=0, le=2, description="QoS for periodic heartbeats.")


class SpoolerConfig(BaseModel):
    """Configuration for local SQLite disk spooler fallback."""
    model_config = ConfigDict(extra="forbid")

    db_path: str = Field(default="data/spooler_queue.db", description="Filesystem path to spooler SQLite file.")
    max_spool_records: int = Field(default=50000, gt=0, description="Maximum queued records before FIFO purge.")


class AuditConfig(BaseModel):
    """Configuration for persistent SQLite audit database."""
    model_config = ConfigDict(extra="forbid")

    db_path: str = Field(default="data/audit_log.db", description="Filesystem path to audit SQLite file.")


class MQTTConfig(BaseModel):
    """Top-level configuration for edge messaging, spooling, and audit logging."""
    model_config = ConfigDict(extra="forbid")

    broker: MQTTBrokerConfig = Field(default_factory=MQTTBrokerConfig)
    topics: MQTTTopicsConfig = Field(default_factory=MQTTTopicsConfig)
    qos: MQTTQoSConfig = Field(default_factory=MQTTQoSConfig)
    spooler: SpoolerConfig = Field(default_factory=SpoolerConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)


class MachineStateScheduleItem(BaseModel):
    """Time window during which a specific machine state applies."""
    model_config = ConfigDict(extra="forbid")

    start: int = Field(..., ge=0, description="Start step index (inclusive).")
    end: int = Field(..., ge=0, description="End step index (exclusive).")
    state: str = Field(..., description="Machine state string (e.g. RUNNING, IDLE, MAINTENANCE, FAULT).")


class InjectedFaultItem(BaseModel):
    """Planned fault injection specification in a scenario."""
    model_config = ConfigDict(extra="forbid")

    fault_type: str = Field(..., description="Fault type identifier.")
    start_step: int = Field(..., ge=0, description="Start step index.")
    duration_steps: int = Field(..., gt=0, description="Duration in steps.")
    intensity: float = Field(default=1.0, gt=0.0, description="Fault intensity multiplier.")
    target_channels: List[str] = Field(default_factory=list, description="Specific target sensor channels.")


class VisionDefectScheduleItem(BaseModel):
    """Time window during which a visual defect is present."""
    model_config = ConfigDict(extra="forbid")

    start: int = Field(..., ge=0, description="Start step index (inclusive).")
    end: int = Field(..., ge=0, description="End step index (exclusive).")


class ScenarioConfig(BaseModel):
    """Standardized scenario workload configuration schema."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique scenario identifier.")
    description: str = Field(..., description="Detailed scenario description.")
    total_steps: int = Field(default=300, gt=0, description="Total simulation steps in scenario.")
    machine_state_schedule: List[MachineStateScheduleItem] = Field(default_factory=list)
    injected_faults: List[InjectedFaultItem] = Field(default_factory=list)
    vision_defect_schedule: List[VisionDefectScheduleItem] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the working directory first, then the repository root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT_DIR / p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load and parse a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing parsed YAML data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If YAML syntax parsing fails.
    """
    file_path = resolve_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {file_path.resolve()}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        logger.error(f"Failed to parse YAML file at {file_path}: {exc}")
        raise ValueError(f"Invalid YAML format in {file_path}: {exc}") from exc


def load_config(path: str | Path, model_cls: Type[T]) -> T:
    """Load, parse, and strictly validate a configuration model from a YAML file.

    Args:
        path: Path to the YAML configuration file.
        model_cls: Pydantic model class to validate against.

    Returns:
        Instantiated and validated Pydantic model instance.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValidationError: If the configuration violates schema constraints.
    """
    data = load_yaml(path)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        logger.error(f"Configuration validation failed for {model_cls.__name__} from {path}:\n{exc}")
        raise


@lru_cache(maxsize=32)
def load_sensor_config(path: str | Path = "configs/sensor_config.yaml") -> SensorConfig:
    """Cached loader for sensor simulation configuration."""
    return load_config(path, SensorConfig)


@lru_cache(maxsize=32)
def load_system_config(path: str | Path = "configs/system_config.yaml") -> SystemConfig:
    """Cached loader for system runtime configuration."""
    return load_config(path, SystemConfig)


@lru_cache(maxsize=32)
def load_policy_config(path: str | Path = "configs/policy_config.yaml") -> PolicyConfig:
    """Cached loader for policy configuration."""
    return load_config(path, PolicyConfig)


@lru_cache(maxsize=32)
def load_mqtt_config(path: str | Path = "configs/mqtt_config.yaml") -> MQTTConfig:
    """Cached loader for MQTT messaging, spooling, and audit configuration."""
    return load_config(path, MQTTConfig)


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    """Loader for standardized scenario workload configuration."""
    return load_config(path, ScenarioConfig)
