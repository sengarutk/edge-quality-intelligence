import inspect
import ast
import numpy as np
import pytest

from src.metrics.operational import (
    compute_quantile_threshold,
    compute_alert_budget_threshold,
    compute_validation_cost_optimal_threshold
)
from src.utils.threshold_lineage import ThresholdRecord, ThresholdLineageAuditor


def test_deployable_threshold_signatures_no_test_labels():
    """
    AST and inspect assertion verifying deployable threshold functions
    strictly do not accept ground-truth labels.
    """
    for fn in [compute_quantile_threshold, compute_alert_budget_threshold]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        for forbidden in ["labels", "y_true", "ground_truth", "gt", "targets"]:
            assert forbidden not in param_names, f"Function {fn.__name__} illegally includes '{forbidden}' in parameters: {param_names}"


def test_threshold_lineage_auditor_clean():
    """
    Verifies that the lineage auditor properly tracks compliant deployable thresholds.
    """
    auditor = ThresholdLineageAuditor()
    rec1 = ThresholdRecord(
        threshold_type="quantile_99",
        source_split="train_normal",
        uses_test_labels=False,
        category="bottle",
        method="patchcore",
        seed=42,
        threshold_value=1.52
    )
    rec2 = ThresholdRecord(
        threshold_type="alert_budget_5",
        source_split="val_nominal",
        uses_test_labels=False,
        category="cable",
        method="padim",
        seed=123,
        threshold_value=2.10
    )
    auditor.record(rec1)
    auditor.record(rec2)
    
    assert auditor.summary()["total_thresholds"] == 2
    assert auditor.summary()["deployable_thresholds"] == 2
    assert len(auditor.audit_leakage()) == 0


def test_threshold_lineage_auditor_catches_leakage():
    """
    Verifies that the lineage auditor catches illegal leakage if test labels were used for deployable thresholds.
    """
    auditor = ThresholdLineageAuditor()
    leaked_rec = ThresholdRecord(
        threshold_type="quantile_99",
        source_split="train_normal",
        uses_test_labels=True,  # Illegal leakage!
        category="bottle",
        method="patchcore",
        seed=42,
        threshold_value=1.52
    )
    auditor.record(leaked_rec)
    violations = auditor.audit_leakage()
    assert len(violations) == 1
    assert violations[0].category == "bottle"


def test_validation_cost_optimal_requires_disjoint_pools():
    """
    Verifies validation cost-optimal threshold derivation from separate pools.
    """
    val_nom = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    val_def = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    tau = compute_validation_cost_optimal_threshold(val_nom, val_def, cost_ratio=10.0, prior=0.01)
    assert 0.4 <= tau <= 0.9