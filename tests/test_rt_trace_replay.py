"""Unit tests for physical sensor trace replay and mixed corruption streams."""

from pathlib import Path
import numpy as np
import pytest

from src.sensor_simulator import MachineState, SensorReading
from src.stream_models import MixedCorruptionStream
from src.trace_replay import RealSensorTraceReplay, generate_sample_physical_trace


def test_trace_generation_and_replay_lifecycle(tmp_path: Path) -> None:
    """Verify trace generation, baseline calibration, and sequential reading playback."""
    csv_file = tmp_path / "test_trace.csv"
    generated_path = generate_sample_physical_trace(csv_path=csv_file, n_steps=200, seed=42)

    assert generated_path.exists()
    assert generated_path.stat().st_size > 0

    replay = RealSensorTraceReplay(
        trace_path=csv_file,
        calibration_window_steps=50,
        z_threshold=3.0,
    )

    assert replay.total_steps() == 200
    assert replay.means["vibration_rms"] > 0.0
    assert replay.stds["vibration_rms"] > 0.0

    # Read steps
    readings = [replay.step() for _ in range(100)]
    assert len(readings) == 100
    assert all(isinstance(r, SensorReading) for r in readings)
    assert all(r.machine_state == MachineState.RUNNING for r in readings[:50])

    # Test reset and generator iteration
    replay.reset()
    all_readings = list(replay)
    assert len(all_readings) == 200

    # Test error handling on missing file
    with pytest.raises(FileNotFoundError):
        RealSensorTraceReplay("non_existent_trace.csv")


def test_mixed_corruption_stream() -> None:
    """Verify stochastic mixed corruption stream applies noise, blur, and compression."""
    rng = np.random.RandomState(42)
    clean_frame = rng.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    # 1. 0% corruption probability (clean pass-through)
    stream_zero = MixedCorruptionStream(p_corrupt=0.0)
    out_clean, applied_zero = stream_zero.corrupt_frame(clean_frame)
    assert np.array_equal(clean_frame, out_clean)
    assert applied_zero == []

    # 2. 100% corruption probability
    stream_full = MixedCorruptionStream(
        p_corrupt=1.0,
        noise_sigma=30.0,
        blur_kernel=9,
        jpeg_quality=20,
        seed=42,
    )
    corrupted, applied = stream_full.corrupt_frame(clean_frame, step=1)

    assert corrupted.shape == clean_frame.shape
    assert corrupted.dtype == np.uint8
    assert len(applied) > 0
    # Must differ from original due to noise / blur / compression
    assert not np.array_equal(clean_frame, corrupted)


def test_synthetic_proxy_traces_replay_through_policies(tmp_path: Path) -> None:
    """The bearing and thermal-creep proxies replay end to end with consistent scoring."""
    from types import SimpleNamespace

    from scripts.run_real_trace_benchmark import replay
    from src.config import PolicyMode
    from src.trace_replay import generate_cmapss_turbofan_trace, generate_ims_bearing_trace

    bank = SimpleNamespace(pools={0: np.full(50, 0.3)}, blur_threshold=100.0)
    for gen, onset in ((generate_ims_bearing_trace, 420), (generate_cmapss_turbofan_trace, 390)):
        path = gen(tmp_path / f"{gen.__name__}.csv")
        base = replay(path, onset, PolicyMode.BASELINE, bank, seed=0)
        full = replay(path, onset, PolicyMode.FULL_POLICY, bank, seed=0)
        # vision alone never sees a sensor fault; the fused policy detects it at onset
        assert base["alerts"] == 0 and base["routing_recall"] == 0.0
        assert full["routing_recall"] == 1.0 and full["n_episodes"] == 1


def test_replay_stops_at_end_of_trace(tmp_path: Path) -> None:
    trace = generate_sample_physical_trace(tmp_path / "t.csv", n_steps=60, seed=1)
    replay = RealSensorTraceReplay(trace, calibration_window_steps=10)
    assert len(list(replay)) == 60
    with pytest.raises(StopIteration):
        replay.step()
