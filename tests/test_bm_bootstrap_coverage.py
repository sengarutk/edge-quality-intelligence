import pytest
from src.metrics.stats import validate_bootstrap_ci_coverage


def test_validate_bootstrap_ci_coverage_nominal_bounds():
    """
    Verifies that Monte Carlo empirical bootstrap coverage matches nominal 95% rate within [0.91, 0.98].
    """
    coverage = validate_bootstrap_ci_coverage(
        n_simulations=200,
        sample_size=60,
        true_mean=10.0,
        true_std=2.0,
        ci=0.95,
        n_bootstraps=300,
        seed=2026
    )

    assert isinstance(coverage, float)
    # Empirical 95% CI coverage for sample size 60 should be in ~[0.91, 0.98]
    assert 0.90 <= coverage <= 0.99