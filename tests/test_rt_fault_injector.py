"""Unit tests for programmatic chaos engineering and fault injector."""

import cv2
import numpy as np
import pytest

from src.fault_injector import ChaosFaultConfig, FaultInjector, FaultType
from src.mqtt_publisher import ResilientMQTTPublisher
from src.sensor_simulator import SensorSimulator


@pytest.fixture
def injector() -> FaultInjector:
    """Fixture providing an initialized FaultInjector."""
    return FaultInjector()


def test_schedule_management(injector: FaultInjector) -> None:
    """Test scheduling, retrieving, and clearing fault schedules."""
    f1 = ChaosFaultConfig(fault_type=FaultType.OPTICAL_BLUR, start_step=10, duration_steps=5, intensity=1.5)
    f2 = ChaosFaultConfig(fault_type=FaultType.NETWORK_PARTITION, start_step=20, duration_steps=10)

    injector.add_fault_schedule(f1)
    injector.add_fault_schedule(f2)

    assert len(injector.get_active_faults(step=5)) == 0
    assert len(injector.get_active_faults(step=12)) == 1
    assert injector.get_active_faults(step=12)[0].fault_type == FaultType.OPTICAL_BLUR
    assert len(injector.get_active_faults(step=25)) == 1

    injector.clear_schedule()
    assert len(injector.get_active_faults(step=12)) == 0


def test_optical_blur_fault_application(injector: FaultInjector) -> None:
    """Test that optical blur fault blurs image and reduces Laplacian variance."""
    f = ChaosFaultConfig(fault_type=FaultType.OPTICAL_BLUR, start_step=0, duration_steps=5, intensity=1.0)
    injector.add_fault_schedule(f)

    # Sharp image with sharp white circle on dark background
    sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(sharp, (50, 50), 30, (255, 255, 255), -1)

    sharp_var = cv2.Laplacian(sharp, cv2.CV_64F).var()
    blurred = injector.apply_optical_fault(sharp, step=2)
    blur_var = cv2.Laplacian(blurred, cv2.CV_64F).var()

    assert blur_var < sharp_var
    assert blur_var < 50.0

    # Inactive step
    unmodified = injector.apply_optical_fault(sharp, step=10)
    assert np.array_equal(unmodified, sharp)


def test_optical_occlusions(injector: FaultInjector) -> None:
    """Test dark and bright optical occlusion fault transformations."""
    injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.OPTICAL_OCCLUSION_DARK, start_step=0, duration_steps=5)
    )
    injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.OPTICAL_OCCLUSION_BRIGHT, start_step=10, duration_steps=5)
    )

    frame = np.full((50, 50, 3), 128, dtype=np.uint8)

    dark = injector.apply_optical_fault(frame, step=2)
    assert np.max(dark) == 0

    bright = injector.apply_optical_fault(frame, step=12)
    assert np.min(bright) == 255


def test_sensor_fault_hooks(injector: FaultInjector) -> None:
    """Test sensor drift and dropout fault hooks."""
    injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.SENSOR_DROPOUT, start_step=0, duration_steps=5, target_channels=["vibration"])
    )
    injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.SENSOR_DRIFT, start_step=10, duration_steps=5)
    )

    sim = SensorSimulator(seed=42)

    # Dropout check
    inject_fault, dropouts = injector.apply_sensor_fault(sim, step=2)
    assert inject_fault is False
    assert dropouts == ["vibration"]

    # Drift is a measurement ramp, not a mechanical fault
    inject_fault, dropouts = injector.apply_sensor_fault(sim, step=12)
    assert inject_fault is False
    assert dropouts is None
    assert 0.0 < injector.get_temperature_drift_c(12) < injector.get_temperature_drift_c(14)
    assert injector.get_temperature_drift_c(20) == 0.0


def test_network_partition_and_vision_shift(injector: FaultInjector, tmp_path) -> None:
    """Test network partition forcing publisher offline and vision distribution shift."""
    injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.NETWORK_PARTITION, start_step=0, duration_steps=5)
    )
    injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.MODEL_DISTRIBUTION_SHIFT, start_step=10, duration_steps=5)
    )

    from src.config import SpoolerConfig
    from src.spooler import DiskSpooler

    pub = ResilientMQTTPublisher(spooler=DiskSpooler(config=SpoolerConfig(db_path=str(tmp_path / "s.db"))))
    pub._is_connected = True

    is_online = injector.apply_network_fault(pub, step=2)
    assert is_online is False
    assert pub.is_connected is False

    assert injector.apply_vision_shift(False, step=12) is True
    assert injector.apply_vision_shift(False, step=25) is False