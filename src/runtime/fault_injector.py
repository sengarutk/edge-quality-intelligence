"""Programmatic Chaos & Fault Injection Subsystem.

Provides controlled simulation of sensor drift, optical blur, camera occlusions,
network partitions, and distribution shifts to validate runtime resilience.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.mqtt_publisher import ResilientMQTTPublisher
from src.sensor_simulator import SensorSimulator


class FaultType(str, Enum):
    """Catalog of supported industrial edge failure modes."""
    NONE = "NONE"
    OPTICAL_BLUR = "OPTICAL_BLUR"
    OPTICAL_OCCLUSION_DARK = "OPTICAL_OCCLUSION_DARK"
    OPTICAL_OCCLUSION_BRIGHT = "OPTICAL_OCCLUSION_BRIGHT"
    SENSOR_DRIFT = "SENSOR_DRIFT"          # gradual temperature measurement drift (no physical fault)
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    MECHANICAL_FAULT = "MECHANICAL_FAULT"  # physical fault: vibration, current and heat all rise
    NETWORK_PARTITION = "NETWORK_PARTITION"
    MODEL_DISTRIBUTION_SHIFT = "MODEL_DISTRIBUTION_SHIFT"


class ChaosFaultConfig(BaseModel):
    """Configuration for a planned chaos fault injection scenario."""
    model_config = ConfigDict(extra="forbid")

    fault_type: FaultType = Field(..., description="Fault mode to inject.")
    start_step: int = Field(..., ge=0, description="Simulation step at which fault activates.")
    duration_steps: int = Field(..., gt=0, description="Duration in steps before fault self-clears.")
    intensity: float = Field(default=1.0, gt=0.0, description="Intensity factor for the fault.")
    target_channels: List[str] = Field(
        default_factory=list, description="Targeted sensor channels (e.g. ['current'] for dropout)."
    )


class FaultInjector:
    """Orchestrator for programmatic edge chaos and failure mode injection."""

    DRIFT_RAMP_C = 12.0  # temperature offset reached at the end of a SENSOR_DRIFT window (x intensity)

    def __init__(self) -> None:
        """Initialize fault injector with an empty schedule."""
        self._schedule: List[ChaosFaultConfig] = []
        self._network_down = False
        logger.info("Initialized FaultInjector.")

    def add_fault_schedule(self, config: ChaosFaultConfig) -> None:
        """Add a planned chaos fault scenario to the schedule.

        Args:
            config: ChaosFaultConfig instance.
        """
        self._schedule.append(config)
        logger.info(
            f"Scheduled fault {config.fault_type.value} from step {config.start_step} "
            f"for {config.duration_steps} steps (intensity={config.intensity})"
        )

    def clear_schedule(self) -> None:
        """Clear all active and scheduled chaos faults."""
        self._schedule.clear()
        logger.info("Cleared all fault injection schedules.")

    def get_active_faults(self, step: int) -> List[ChaosFaultConfig]:
        """Return all faults active at the given simulation step.

        Args:
            step: Current simulation step index.

        Returns:
            List of active ChaosFaultConfig instances.
        """
        return [
            f for f in self._schedule
            if f.start_step <= step < (f.start_step + f.duration_steps)
        ]

    def apply_optical_fault(self, frame: np.ndarray, step: int) -> np.ndarray:
        """Apply active optical degradation faults to the camera frame.

        Args:
            frame: Input camera frame (uint8).
            step: Current simulation step index.

        Returns:
            Transformed or original frame.
        """
        active = self.get_active_faults(step)
        for fault in active:
            if fault.fault_type == FaultType.OPTICAL_BLUR:
                ksize = int(35 * fault.intensity)
                if ksize % 2 == 0:
                    ksize += 1
                ksize = max(3, ksize)
                sigma = 20.0 * fault.intensity
                return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=sigma)

            elif fault.fault_type == FaultType.OPTICAL_OCCLUSION_DARK:
                return np.zeros_like(frame)

            elif fault.fault_type == FaultType.OPTICAL_OCCLUSION_BRIGHT:
                return np.full_like(frame, 255)

        return frame

    def apply_sensor_fault(
        self, simulator: Optional[SensorSimulator], step: int
    ) -> Tuple[bool, Optional[List[str]]]:
        """Return (inject_mechanical_fault, dropout_channels) for this step.

        SENSOR_DROPOUT without explicit target channels drops the current channel.
        SENSOR_DRIFT is *not* a physical fault; use :meth:`get_temperature_drift_c` for it.
        """
        inject_fault = False
        dropouts: List[str] = []
        for fault in self.get_active_faults(step):
            if fault.fault_type == FaultType.SENSOR_DROPOUT:
                dropouts.extend(fault.target_channels or ["current"])
            elif fault.fault_type == FaultType.MECHANICAL_FAULT:
                inject_fault = True
        return inject_fault, (sorted(set(dropouts)) if dropouts else None)

    def get_temperature_drift_c(self, step: int) -> float:
        """Additive temperature measurement drift: a linear ramp over each SENSOR_DRIFT window."""
        offset = 0.0
        for fault in self.get_active_faults(step):
            if fault.fault_type == FaultType.SENSOR_DRIFT:
                progress = (step - fault.start_step + 1) / fault.duration_steps
                offset += self.DRIFT_RAMP_C * fault.intensity * progress
        return offset

    def is_network_partitioned(self, step: int) -> bool:
        return any(f.fault_type == FaultType.NETWORK_PARTITION for f in self.get_active_faults(step))

    def apply_network_fault(self, publisher: ResilientMQTTPublisher, step: int) -> bool:
        """Disconnect or reconnect the publisher at partition boundaries.

        Uses the publisher's public pause/resume API, which closes and reopens the real
        broker connection. Returns True while the network is available.
        """
        down = self.is_network_partitioned(step)
        if down and not self._network_down:
            publisher.pause_network()
        elif not down and self._network_down:
            publisher.resume_network()
        self._network_down = down
        return not down

    def apply_vision_shift(self, inject_anomaly: bool, step: int) -> bool:
        """Force visual defect injection during model distribution shift faults.

        Args:
            inject_anomaly: Base anomaly injection boolean.
            step: Current simulation step index.

        Returns:
            True if model distribution shift fault is active.
        """
        active = self.get_active_faults(step)
        for fault in active:
            if fault.fault_type == FaultType.MODEL_DISTRIBUTION_SHIFT:
                return True
        return inject_anomaly