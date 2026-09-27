"""Reproducibility, threshold governance manifest, and non-IID stream model tests."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.stream_models import (
    MarkovianDefectGenerator,
    PoissonBurstDefectGenerator,
    ThermalDriftProfile,
)


def test_threshold_manifest_matches_runtime_config() -> None:
    """The documented parameter manifest must describe the configuration actually used."""
    from src.config import load_policy_config

    data = json.loads(Path("configs/threshold_manifest.json").read_text(encoding="utf-8"))
    cfg = load_policy_config()
    p = data["parameters"]
    assert p["vision_high"]["value"] == cfg.thresholds.vision_high
    assert p["vision_medium"]["value"] == cfg.thresholds.vision_medium
    assert p["sensor_anomaly"]["value"] == cfg.thresholds.sensor_anomaly
    assert p["cross_modal_divergence"]["value"] == cfg.thresholds.cross_modal_divergence
    assert p["required_k"]["value"] == cfg.confirmation_window.required_k
    assert p["window_size_n"]["value"] == cfg.confirmation_window.window_size_n
    assert p["cooldown_steps"]["value"] == cfg.cooldown.cooldown_steps
    assert p["alpha_vision"]["value"] == cfg.temporal_smoothing.alpha_vision
    assert p["alpha_sensor"]["value"] == cfg.temporal_smoothing.alpha_sensor


def test_deterministic_stream_reproducibility() -> None:
    """Verify deterministic reproducibility of non-IID temporal stream models."""
    gen1 = MarkovianDefectGenerator(seed=2026)
    seq1 = gen1.generate_sequence(100)

    gen2 = MarkovianDefectGenerator(seed=2026)
    seq2 = gen2.generate_sequence(100)

    assert seq1 == seq2
    assert any(seq1)  # Contains defect blocks
    assert not all(seq1)  # Contains nominal blocks

    # Poisson burst generator reproducibility
    p1 = PoissonBurstDefectGenerator(seed=2026)
    p_seq1 = p1.generate_sequence(100)

    p2 = PoissonBurstDefectGenerator(seed=2026)
    p_seq2 = p2.generate_sequence(100)

    assert p_seq1 == p_seq2


def test_thermal_drift_profile() -> None:
    """Verify thermal drift profile calculations across linear and exponential modes."""
    linear_prof = ThermalDriftProfile(base_temp_c=40.0, peak_temp_c=80.0, drift_type="linear", ramp_steps=100)
    t_start = linear_prof.get_temperature(0, noise_std=0.0)
    t_mid = linear_prof.get_temperature(50, noise_std=0.0)
    t_end = linear_prof.get_temperature(100, noise_std=0.0)

    assert np.isclose(t_start, 40.0)
    assert np.isclose(t_mid, 60.0)
    assert np.isclose(t_end, 80.0)

    exp_prof = ThermalDriftProfile(base_temp_c=40.0, peak_temp_c=80.0, drift_type="exponential", ramp_steps=100)
    t_exp_mid = exp_prof.get_temperature(50, noise_std=0.0)
    # Exponential ramp accelerates later, so midpoint should be lower than linear midpoint
    assert t_exp_mid < t_mid
