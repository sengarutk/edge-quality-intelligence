import numpy as np
import pytest

from src.experiments.operational_eval import ProductionStreamSimulator


def test_iid_stream_generation():
    """
    Verifies IID stream simulation matches target length and defect prior.
    """
    nom = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    defs = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    sim = ProductionStreamSimulator(nom, defs, seed=42)
    
    labels, scores = sim.simulate_iid_stream(n_total=10000, defect_prior=0.05)
    assert len(labels) == 10000
    assert len(scores) == 10000
    empirical_prior = np.mean(labels)
    assert 0.04 <= empirical_prior <= 0.06
    assert np.all(np.isfinite(scores))


def test_block_correlated_burst_stream():
    """
    Verifies Two-State Markov chain generates expected burst transitions and defect clusters.
    """
    nom = np.array([0.1, 0.2, 0.3])
    defs = np.array([1.1, 1.2, 1.3])
    sim = ProductionStreamSimulator(nom, defs, seed=123)
    
    labels, scores = sim.simulate_block_correlated_stream(n_total=10000, defect_prior=0.05, mean_block_length=20)
    assert len(labels) == 10000
    assert len(scores) == 10000
    assert np.all(np.isfinite(scores))
    
    # Calculate defect run lengths (consecutive 1s)
    defect_runs = []
    curr_run = 0
    for y in labels:
        if y == 1:
            curr_run += 1
        else:
            if curr_run > 0:
                defect_runs.append(curr_run)
                curr_run = 0
    if curr_run > 0:
        defect_runs.append(curr_run)
        
    if len(defect_runs) > 0:
        mean_run = np.mean(defect_runs)
        # Should be much higher than IID geometric run length (which is ~1.05 for prior 0.05)
        assert mean_run > 3.0


def test_drift_stream():
    """
    Verifies gradual drift generator increases nominal anomaly scores monotonically over time.
    """
    nom = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    defs = np.array([1.1, 1.2, 1.3])
    sim = ProductionStreamSimulator(nom, defs, seed=2026)
    
    labels, scores = sim.simulate_drift_stream(n_total=10000, defect_prior=0.01, drift_slope=0.50)
    assert len(labels) == 10000
    assert len(scores) == 10000
    
    # First 1000 nominal scores vs last 1000 nominal scores
    nom_early = scores[:1000][labels[:1000] == 0]
    nom_late = scores[-1000:][labels[-1000:] == 0]
    if len(nom_early) > 0 and len(nom_late) > 0:
        assert np.mean(nom_late) > np.mean(nom_early)