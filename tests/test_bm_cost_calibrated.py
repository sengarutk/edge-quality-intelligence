import os
import numpy as np
import pytest

from src.metrics.cost_calibrated import (
    CostCalibratedThresholdOptimizer,
    compute_empirical_cost_curve,
    optimize_cct_threshold
)
from src.experiments.cct_ablation import (
    stratified_split_50_50,
    run_cct_out_of_sample_ablation
)


def test_compute_empirical_cost_curve_convexity():
    np.random.seed(42)
    nom = np.random.normal(0.2, 0.05, 500)
    defc = np.random.normal(0.8, 0.05, 100)
    scores = np.concatenate([nom, defc])
    labels = np.concatenate([np.zeros(500, dtype=int), np.ones(100, dtype=int)])

    taus, costs, fprs, fnrs = compute_empirical_cost_curve(scores, labels, cost_ratio=10.0, prior=0.01)

    assert len(taus) == 200
    assert len(costs) == 200
    assert len(fprs) == 200
    assert len(fnrs) == 200

    assert np.all(np.diff(fprs) <= 1e-6)
    assert np.all(np.diff(fnrs) >= -1e-6)

    min_idx = np.argmin(costs)
    assert 0 < min_idx < len(taus) - 1


def test_optimize_cct_threshold_budget_constraint():
    np.random.seed(42)
    nom = np.random.normal(0.2, 0.05, 1000)
    defc = np.random.normal(0.8, 0.05, 100)
    scores = np.concatenate([nom, defc])
    labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(100, dtype=int)])

    res = optimize_cct_threshold(scores, labels, cost_ratio=10.0, prior=0.01, max_alerts_per_1k=5.0)

    assert "threshold" in res
    assert "min_expected_cost" in res
    assert "achieved_val_fpr" in res
    assert "budget_satisfied" in res

    assert res["achieved_val_fpr"] <= 0.005 + 1e-4
    assert res["budget_satisfied"] is True


def test_stratified_split_50_50():
    labels = np.array([0]*100 + [1]*20)
    scores = np.random.rand(120)

    cal_s, cal_y, ev_s, ev_y = stratified_split_50_50(labels, scores, seed=42)
    assert len(cal_s) == 60
    assert len(ev_s) == 60
    assert np.sum(cal_y == 1) == 10
    assert np.sum(ev_y == 1) == 10


def test_run_cct_out_of_sample_ablation_synthetic(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)

    np.savez(
        scores_dir / "bottle_patchcore_42.npz",
        image_labels=np.array([0]*80 + [1]*20),
        image_scores=np.concatenate([np.random.normal(0.2, 0.05, 80), np.random.normal(0.8, 0.05, 20)])
    )

    df = run_cct_out_of_sample_ablation(
        scores_dir=str(scores_dir),
        output_dir=str(out_dir)
    )

    assert len(df) == 1
    assert "cwe_cct_r10" in df.columns
    assert "cwe_q99_r10" in df.columns
    assert "cwe_b5_r10" in df.columns


def test_cct_edge_cases():
    res_empty = optimize_cct_threshold(np.array([]), np.array([]))
    assert res_empty["threshold"] == 0.0
    assert res_empty["min_expected_cost"] == 0.0

    nom_only = np.array([0.1, 0.2, 0.3])
    y_nom = np.array([0, 0, 0])
    res_nom = optimize_cct_threshold(nom_only, y_nom, max_alerts_per_1k=5.0)
    assert res_nom["threshold"] >= 0.1

    def_only = np.array([0.7, 0.8, 0.9])
    y_def = np.array([1, 1, 1])
    res_def = optimize_cct_threshold(def_only, y_def, max_alerts_per_1k=5.0)
    assert res_def["threshold"] >= 0.0