"""Metrics package: detection, operational, statistical and queueing metrics."""

from src.metrics.image_metrics import (
    compute_image_auroc,
    compute_image_ap,
    compute_optimal_f1,
    compute_optimal_f1_threshold,
    compute_quantile_threshold,
)
from src.metrics.pixel_metrics import (
    compute_pixel_auroc,
    compute_pixel_ap,
    compute_aupro,
)
from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error,
    compute_alert_budget_threshold,
    compute_validation_cost_optimal_threshold,
    compute_tpr_at_alert_budget,
    compute_operator_overload,
)
from src.metrics.cost_calibrated import (
    CostCalibratedThresholdOptimizer,
    compute_empirical_cost_curve,
    optimize_cct_threshold,
)
from src.metrics.stats import (
    bootstrap_ci,
    compute_bootstrap_ci,
    paired_wilcoxon_test,
    holm_bonferroni_correction,
    validate_bootstrap_ci_coverage,
    hierarchical_bootstrap_ci,
    compute_paired_wilcoxon_analysis,
    compute_wilcoxon_significance,
    holm_bonferroni_from_pvalues,
)
from src.metrics.significance import apply_holm_bonferroni_correction, paired_significance_test
from src.metrics.stream import compute_stream_metrics, episodes_from_steps
from src.metrics.calibration import (
    compute_ece,
    fit_isotonic_calibration,
    get_reliability_diagram_data,
)
from src.metrics.reference_aupro import compute_aupro_reference

from src.metrics.queue_model import OperatorQueueModel
from src.metrics.evaluator import BenchmarkEvaluator, aggregate_ablation_results

__all__ = [
    "compute_image_auroc",
    "compute_image_ap",
    "compute_optimal_f1",
    "compute_optimal_f1_threshold",
    "compute_quantile_threshold",
    "compute_pixel_auroc",
    "compute_pixel_ap",
    "compute_aupro",
    "compute_fa_at_1k",
    "compute_md_at_1k",
    "compute_cost_weighted_error",
    "compute_alert_budget_threshold",
    "compute_validation_cost_optimal_threshold",
    "compute_tpr_at_alert_budget",
    "compute_operator_overload",
    "CostCalibratedThresholdOptimizer",
    "compute_empirical_cost_curve",
    "optimize_cct_threshold",
    "bootstrap_ci",
    "compute_bootstrap_ci",
    "paired_wilcoxon_test",
    "holm_bonferroni_correction",
    "validate_bootstrap_ci_coverage",
    "hierarchical_bootstrap_ci",
    "compute_paired_wilcoxon_analysis",
    "compute_wilcoxon_significance",
    "apply_holm_bonferroni_correction",
    "compute_ece",
    "fit_isotonic_calibration",
    "get_reliability_diagram_data",
    "compute_aupro_reference",
    "OperatorQueueModel",
    "BenchmarkEvaluator",
    "aggregate_ablation_results",
    "holm_bonferroni_from_pvalues",
    "paired_significance_test",
    "compute_stream_metrics",
    "episodes_from_steps",
]
