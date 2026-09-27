"""Tests for the operator review queue models."""

import numpy as np
import pytest

from src.metrics.queue_model import OperatorQueueModel


def test_mm1_closed_forms():
    q = OperatorQueueModel(60.0)
    m = q.analyze_mm1(12.0)
    assert m["stable"] and m["utilization"] == pytest.approx(0.2)
    assert m["mean_queue_length"] == pytest.approx(0.05)
    assert m["mean_wait_time_minutes"] == pytest.approx(0.25)
    assert m["p_at_least_10_in_system"] == pytest.approx(0.2 ** 10)
    assert q.analyze_mm1(0.0)["mean_queue_length"] == 0.0


@pytest.mark.parametrize("lam", [60.0, 75.0])
def test_unstable_regime(lam):
    m = OperatorQueueModel(60.0).analyze_mm1(lam)
    assert not m["stable"] and m["mean_queue_length"] == float("inf")


def test_md1_halves_mm1_queue():
    q = OperatorQueueModel(60.0)
    assert q.analyze_md1(30.0)["mean_queue_length"] == pytest.approx(0.5 * q.analyze_mm1(30.0)["mean_queue_length"])
    assert q.analyze_md1(30.0)["p_at_least_10_in_system"] is None


def test_simulation_matches_pollaczek_khinchine():
    """Log-normal service with sigma: CV^2 = exp(sigma^2) - 1; L_q = rho^2 (1 + CV^2) / (2 (1 - rho))."""
    q = OperatorQueueModel(60.0)
    sigma, lam = 0.6, 30.0
    rho, cv2 = lam / 60.0, np.exp(sigma ** 2) - 1
    expected = rho ** 2 * (1 + cv2) / (2 * (1 - rho))
    sims = [q.simulate_variable_service(lam, duration_hours=200, service_sigma=sigma, seed=s)["mean_queue_length"]
            for s in range(4)]
    assert np.mean(sims) == pytest.approx(expected, rel=0.1)


def test_simulation_exposes_instability():
    q = OperatorQueueModel(60.0)
    short = q.simulate_variable_service(90.0, duration_hours=2, seed=1)["backlog_at_end"]
    long = q.simulate_variable_service(90.0, duration_hours=8, seed=1)["backlog_at_end"]
    assert long > short > 0


def test_replay_of_recorded_arrivals():
    q = OperatorQueueModel(60.0)
    res = q.simulate_variable_service(0.0, duration_hours=1, arrival_times_min=np.array([0.0, 0.1, 0.2]), seed=0)
    assert res["n_arrivals"] == 3 and res["max_wait_minutes"] > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        OperatorQueueModel(0.0)
    with pytest.raises(ValueError):
        OperatorQueueModel(60.0).analyze_mm1(-1.0)
