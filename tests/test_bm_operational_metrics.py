import numpy as np
import pytest

from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error,
    compute_tpr_at_alert_budget,
    compute_operator_overload
)
from src.metrics.stats import bootstrap_ci, compute_wilcoxon_significance


def test_perfect_classifier():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95])
    threshold = 0.5

    fa = compute_fa_at_1k(labels, scores, threshold)
    md = compute_md_at_1k(labels, scores, threshold)
    cwe = compute_cost_weighted_error(labels, scores, threshold, cost_ratio=10.0)

    assert fa == 0.0
    assert md == 0.0
    assert cwe == 0.0


def test_classifier_extremes():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # All-positive predictions (threshold <= 0.1)
    fa_all_pos = compute_fa_at_1k(labels, scores, threshold=0.1)
    md_all_pos = compute_md_at_1k(labels, scores, threshold=0.1)
    assert fa_all_pos == 1000.0
    assert md_all_pos == 0.0

    # All-negative predictions (threshold >= 1.0)
    fa_all_neg = compute_fa_at_1k(labels, scores, threshold=1.0)
    md_all_neg = compute_md_at_1k(labels, scores, threshold=1.0)
    assert fa_all_neg == 0.0
    assert md_all_neg == 1000.0


def test_cost_weighted_error_manual_parity():
    # 4 nominals (indices 0..3), 4 defects (indices 4..7)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.3, 0.4, 0.8, 0.9, 0.7])
    # At threshold 0.5:
    # Nominals: [0.1, 0.6, 0.2, 0.3] -> 1 False Alarm (index 1: 0.6 >= 0.5) -> FP = 1
    # Defects: [0.4, 0.8, 0.9, 0.7] -> 1 Missed Defect (index 4: 0.4 < 0.5) -> FN = 1
    # Total samples N = 8
    # With r = 10.0: Total Cost = (1 * 1.0) + (1 * 10.0) = 11.0 -> CWE = 11.0 / 8 = 1.375
    # With r = 50.0: Total Cost = (1 * 1.0) + (1 * 50.0) = 51.0 -> CWE = 51.0 / 8 = 6.375
    cwe_10 = compute_cost_weighted_error(labels, scores, threshold=0.5, cost_ratio=10.0)
    cwe_50 = compute_cost_weighted_error(labels, scores, threshold=0.5, cost_ratio=50.0)

    assert np.isclose(cwe_10, 1.375)
    assert np.isclose(cwe_50, 6.375)


def test_tpr_at_alert_budget():
    np.random.seed(42)
    # 1000 normal items, 100 defective items
    norm_scores = np.random.normal(0.2, 0.05, 1000)
    def_scores = np.random.normal(0.8, 0.05, 100)

    labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(100, dtype=int)])
    scores = np.concatenate([norm_scores, def_scores])

    res = compute_tpr_at_alert_budget(labels, scores, max_alerts_per_1k=5.0)

    assert "tpr" in res
    assert "fpr" in res
    assert "md_at_1k" in res
    assert "threshold" in res
    assert "fa_at_1k" in res

    # FA@1k should be within target bound (~5.0 / 1000)
    assert res["fa_at_1k"] <= 6.0
    assert res["tpr"] >= 0.95


def test_operator_overload():
    # 3000 parts -> 3 windows of 1000 parts
    # Window 0: 20 alerts (<= 60 -> not overloaded)
    # Window 1: 80 alerts (> 60 -> overloaded)
    # Window 2: 10 alerts (<= 60 -> not overloaded)
    w0 = [1]*20 + [0]*980
    w1 = [1]*80 + [0]*920
    w2 = [1]*10 + [0]*990

    alert_stream = np.array(w0 + w1 + w2)
    res = compute_operator_overload(alert_stream, operator_capacity_per_window=60, window_size=1000)

    assert res["num_windows"] == 3
    assert np.isclose(res["mean_load"], (20 + 80 + 10) / 3.0)
    assert res["peak_load"] == 80.0
    assert np.isclose(res["overload_probability"], 1.0 / 3.0)


def test_stats_bootstrap_and_wilcoxon():
    values = np.array([0.95, 0.96, 0.94, 0.97, 0.95])
    mean, low, high = bootstrap_ci(values, n_bootstraps=200, seed=42)
    assert low <= mean <= high

    # Wilcoxon paired test
    a = np.array([0.95, 0.96, 0.98, 0.97, 0.99])
    b = np.array([0.80, 0.82, 0.85, 0.81, 0.84])
    w_res = compute_wilcoxon_significance(a, b)
    assert "statistic" in w_res
    assert "p_value" in w_res
    assert w_res["p_value"] < 0.10


def test_edge_cases_empty_inputs():
    assert compute_fa_at_1k(np.array([]), np.array([]), threshold=0.5) == 0.0
    assert compute_md_at_1k(np.array([]), np.array([]), threshold=0.5) == 0.0
    assert compute_cost_weighted_error(np.array([]), np.array([]), threshold=0.5) == 0.0
    
    empty_tpr = compute_tpr_at_alert_budget(np.array([]), np.array([]), max_alerts_per_1k=5.0)
    assert empty_tpr["tpr"] == 0.0

    empty_ovl = compute_operator_overload(np.array([]))
    assert empty_ovl["mean_load"] == 0.0
    assert empty_ovl["num_windows"] == 0
