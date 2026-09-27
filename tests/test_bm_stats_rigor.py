import numpy as np
import pytest

from src.metrics.stats import (
    bootstrap_ci,
    hierarchical_bootstrap_ci,
    compute_paired_wilcoxon_analysis,
    holm_bonferroni_from_pvalues
)


def test_hierarchical_bootstrap_ci():
    """
    Verifies two-stage hierarchical resampling correctly computes non-degenerate bounds.
    """
    rng = np.random.RandomState(42)
    # Generate 5 synthetic runs, each with 100 items
    records = []
    for r in range(5):
        scores = rng.normal(loc=1.0 + 0.1 * r, scale=0.2, size=100)
        labels = (rng.rand(100) < 0.2).astype(int)
        records.append({"scores": scores, "labels": labels})

    def metric_fn(recs):
        all_means = [np.mean(r["scores"]) for r in recs]
        return float(np.mean(all_means))

    res = hierarchical_bootstrap_ci(records, metric_fn, n_resamples=500, ci=0.95, seed=42)
    assert "estimate" in res
    assert "ci_low" in res
    assert "ci_high" in res
    assert res["ci_low"] <= res["estimate"] <= res["ci_high"]
    assert res["ci_high"] > res["ci_low"]
    assert res["bootstrap_unit"] == "hierarchical_item_run"


def test_compute_paired_wilcoxon_analysis():
    """
    Verifies Wilcoxon statistics, Hodges-Lehmann median difference, and rank-biserial correlation.
    """
    # Method A consistently better than Method B
    method_a = np.array([0.05, 0.10, 0.08, 0.12, 0.09, 0.07, 0.11, 0.06])
    method_b = np.array([0.50, 0.60, 0.45, 0.70, 0.55, 0.40, 0.65, 0.48])

    res = compute_paired_wilcoxon_analysis(method_a, method_b, alpha=0.05)
    assert res["n_pairs"] == 8
    assert res["p_value"] < 0.05
    assert res["mean_diff"] < 0  # Method A is lower (better CWE)
    assert res["hodges_lehmann"] < 0
    assert res["rank_biserial"] == -1.0  # Perfect negative rank biserial correlation


def test_holm_bonferroni_from_pvalues():
    """
    Verifies step-down Holm-Bonferroni correction maintains strict FWER bounds.
    """
    p_vals = {
        "test_1": 0.001,
        "test_2": 0.015,
        "test_3": 0.040,
        "test_4": 0.250
    }
    corrected = holm_bonferroni_from_pvalues(p_vals, alpha=0.05)
    assert len(corrected) == 4
    # test_1 (rank 1): alpha_1 = 0.05 / 4 = 0.0125 -> Significant
    assert corrected["test_1"]["is_significant"] is True
    assert corrected["test_1"]["rank"] == 1
    # test_4 should definitely not be significant
    assert corrected["test_4"]["is_significant"] is False
    # Adjusted p-values must be non-decreasing with rank
    ranks = sorted(corrected.values(), key=lambda x: x["rank"])
    for i in range(len(ranks) - 1):
        assert ranks[i]["adjusted_p"] <= ranks[i + 1]["adjusted_p"]