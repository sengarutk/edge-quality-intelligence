import numpy as np
import pytest

from src.experiments.operational_eval import ProductionStreamSimulator


def test_production_stream_simulator_priors():
    np.random.seed(42)
    nom_scores = np.random.normal(0.2, 0.05, 500)
    def_scores = np.random.normal(0.8, 0.05, 100)

    sim = ProductionStreamSimulator(nom_scores, def_scores, seed=42)

    for prior in [0.01, 0.05, 0.15]:
        labels, scores = sim.simulate_stream(n_total=10000, defect_prior=prior)
        assert len(labels) == 10000
        assert len(scores) == 10000
        assert set(np.unique(labels)).issubset({0, 1})

        n_defect = np.sum(labels == 1)
        expected_defect = int(round(10000 * prior))
        assert n_defect == expected_defect


def test_evaluate_threshold_strategies():
    np.random.seed(42)
    nom_scores = np.random.normal(0.2, 0.05, 500)
    def_scores = np.random.normal(0.8, 0.05, 100)

    sim = ProductionStreamSimulator(nom_scores, def_scores, seed=42)
    labels, scores = sim.simulate_stream(n_total=5000, defect_prior=0.05)

    strat_res = sim.evaluate_threshold_strategies(
        stream_labels=labels,
        stream_scores=scores,
        nominal_ref_scores=nom_scores,
        cost_ratio=10.0,
        defect_prior=0.05
    )

    for strat_key in ["oracle_f1", "nominal_quantile_99", "cost_optimal"]:
        assert strat_key in strat_res
        metrics = strat_res[strat_key]
        for k in ["threshold", "fa_at_1k", "md_at_1k", "cwe", "tpr"]:
            assert k in metrics
            assert not np.isnan(metrics[k])
            assert not np.isinf(metrics[k])


def test_single_class_or_empty_stream():
    # Empty nominal pool
    sim_no_nom = ProductionStreamSimulator(nominal_scores=np.array([]), defect_scores=np.array([0.8, 0.9]))
    labels, scores = sim_no_nom.simulate_stream(n_total=100, defect_prior=0.1)
    assert len(labels) == 100
    assert len(scores) == 100

    # Empty defect pool
    sim_no_def = ProductionStreamSimulator(nominal_scores=np.array([0.1, 0.2]), defect_scores=np.array([]))
    labels2, scores2 = sim_no_def.simulate_stream(n_total=100, defect_prior=0.1)
    assert len(labels2) == 100
    assert len(scores2) == 100



def test_mixed_corruption_stream_simulator():
    np.random.seed(42)
    nom_clean = np.random.normal(0.2, 0.05, 300)
    def_clean = np.random.normal(0.8, 0.05, 100)
    nom_corr = np.random.normal(0.35, 0.08, 300)
    def_corr = np.random.normal(0.65, 0.08, 100)

    from src.experiments.operational_eval import MixedCorruptionStreamSimulator
    sim = MixedCorruptionStreamSimulator(nom_clean, def_clean, nom_corr, def_corr, seed=2026)

    labels, scores, is_corr = sim.simulate_mixed_stream(n_total=5000, defect_prior=0.02, corruption_prob=0.20)
    assert len(labels) == 5000
    assert len(scores) == 5000
    assert len(is_corr) == 5000

    corr_fraction = np.mean(is_corr)
    assert 0.16 <= corr_fraction <= 0.24  # Around 20%

    metrics = sim.evaluate_mixed_stream(labels, scores, tau=0.5, cost_ratio=10.0, prior=0.02)
    assert "fa_at_1k" in metrics
    assert "md_at_1k" in metrics
    assert "cwe" in metrics
    assert 0.0 <= metrics["fa_at_1k"] <= 1000.0
    assert 0.0 <= metrics["md_at_1k"] <= 1000.0