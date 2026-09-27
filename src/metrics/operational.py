from typing import Dict, Any, Tuple, Optional
import numpy as np


def compute_fa_at_1k(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """
    Computes False Alarms per 1,000 nominal parts at a given decision threshold.
    FA@1k = (FP / Total Nominal) * 1000.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    nom_mask = (labels == 0)
    n_nom = np.sum(nom_mask)
    if n_nom == 0:
        return 0.0
    fp = np.sum(scores[nom_mask] >= threshold)
    return float((fp / n_nom) * 1000.0)


def compute_md_at_1k(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """
    Computes Missed Defects per 1,000 defective parts at a given decision threshold.
    MD@1k = (FN / Total Defective) * 1000.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    def_mask = (labels == 1)
    n_def = np.sum(def_mask)
    if n_def == 0:
        return 0.0
    fn = np.sum(scores[def_mask] < threshold)
    return float((fn / n_def) * 1000.0)


def compute_cost_weighted_error(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cost_ratio: float = 10.0,
    prior: Optional[float] = None
) -> float:
    """
    Computes Asymmetric Cost-Weighted Error (CWE) per inspected part.
    If prior is provided (e.g. 0.01 for factory streams):
      CWE = (1 - prior) * FPR * 1.0 + prior * FNR * cost_ratio
    Otherwise:
      CWE = (FP + cost_ratio * FN) / Total Items.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n_total = len(labels)
    if n_total == 0:
        return 0.0

    if prior is not None:
        nom_mask = (labels == 0)
        def_mask = (labels == 1)
        n_nom = np.sum(nom_mask)
        n_def = np.sum(def_mask)
        fpr = float(np.sum(scores[nom_mask] >= threshold) / n_nom) if n_nom > 0 else 0.0
        fnr = float(np.sum(scores[def_mask] < threshold) / n_def) if n_def > 0 else 0.0
        return float((1.0 - prior) * fpr * 1.0 + prior * fnr * cost_ratio)

    fp = np.sum((labels == 0) & (scores >= threshold))
    fn = np.sum((labels == 1) & (scores < threshold))
    return float((fp + cost_ratio * fn) / n_total)


def compute_quantile_threshold(nominal_scores: np.ndarray, quantile: float = 0.99) -> float:
    """
    Deployable Leakage-Free Threshold derivation: computes empirical quantile threshold
    strictly over nominal/reference scores.
    """
    nominal_scores = np.asarray(nominal_scores, dtype=np.float64).ravel()
    if len(nominal_scores) == 0:
        return 0.0
    return float(np.quantile(nominal_scores, quantile))


def compute_alert_budget_threshold(nominal_scores: np.ndarray, max_alerts_per_1k: float = 5.0) -> float:
    """
    Deployable Leakage-Free Threshold derivation: derives threshold strictly from nominal
    scores such that expected false alert rate is bounded by max_alerts_per_1k.
    """
    nominal_scores = np.asarray(nominal_scores, dtype=np.float64).ravel()
    if len(nominal_scores) == 0:
        return 0.0
    allowed_fpr = max_alerts_per_1k / 1000.0
    target_quantile = max(0.0, min(1.0, 1.0 - allowed_fpr))
    return float(np.quantile(nominal_scores, target_quantile))


def compute_validation_cost_optimal_threshold(
    val_nominal_scores: np.ndarray,
    val_defect_scores: np.ndarray,
    cost_ratio: float = 10.0,
    prior: float = 0.01
) -> float:
    """
    Deployable Leakage-Free Threshold: derives cost-optimal threshold from separate validation pools.
    Minimizes expected risk: (1 - prior) * P(S >= tau | Nominal) + prior * cost_ratio * P(S < tau | Defect).
    """
    val_nominal_scores = np.asarray(val_nominal_scores, dtype=np.float64).ravel()
    val_defect_scores = np.asarray(val_defect_scores, dtype=np.float64).ravel()

    all_scores = np.sort(np.unique(np.concatenate([val_nominal_scores, val_defect_scores])))
    if len(all_scores) == 0:
        return 0.0

    best_tau = float(all_scores[0])
    min_cost = float("inf")

    n_nom = len(val_nominal_scores)
    n_def = len(val_defect_scores)

    for tau in all_scores:
        fpr = np.sum(val_nominal_scores >= tau) / n_nom if n_nom > 0 else 0.0
        fnr = np.sum(val_defect_scores < tau) / n_def if n_def > 0 else 0.0
        expected_cost = (1.0 - prior) * fpr + prior * cost_ratio * fnr
        if expected_cost < min_cost:
            min_cost = expected_cost
            best_tau = float(tau)

    return best_tau


def compute_tpr_at_alert_budget(
    labels: np.ndarray,
    scores: np.ndarray,
    max_alerts_per_1k: float = 5.0
) -> Dict[str, float]:
    """
    Evaluates True Positive Rate (Defect Recall) when the false alert rate on nominal parts
    is bounded by max_alerts_per_1k (e.g. 5 alarms per 1k normal items -> FPR <= 0.005).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    nom_scores = scores[labels == 0]
    def_scores = scores[labels == 1]

    if len(nom_scores) == 0 or len(def_scores) == 0:
        return {
            "tpr": 0.0,
            "fpr": 0.0,
            "md_at_1k": 0.0,
            "fa_at_1k": 0.0,
            "threshold": 0.0,
            "actual_fa_per_1k": 0.0,
            "max_alerts_per_1k": max_alerts_per_1k
        }

    tau = compute_alert_budget_threshold(nom_scores, max_alerts_per_1k=max_alerts_per_1k)
    tpr = float(np.sum(def_scores >= tau) / len(def_scores))
    fpr = float(np.sum(nom_scores >= tau) / len(nom_scores))
    fa_1k = fpr * 1000.0
    md_1k = (1.0 - tpr) * 1000.0

    return {
        "tpr": tpr,
        "fpr": fpr,
        "md_at_1k": md_1k,
        "fa_at_1k": fa_1k,
        "threshold": tau,
        "actual_fa_per_1k": fa_1k,
        "max_alerts_per_1k": max_alerts_per_1k
    }


def compute_operator_overload(
    alert_stream: np.ndarray,
    operator_capacity_per_window: int = 60,
    window_size: int = 1000
) -> Dict[str, float]:
    """
    Analyzes queue congestion and review overload for human inspectors.
    Supports both non-overlapping window blocks and sliding windows.
    """
    alert_stream = np.asarray(alert_stream, dtype=int)
    n = len(alert_stream)
    if n == 0 or window_size <= 0:
        return {
            "p_overload": 0.0,
            "overload_probability": 0.0,
            "mean_window_alerts": 0.0,
            "mean_load": 0.0,
            "max_window_alerts": 0.0,
            "peak_load": 0.0,
            "total_alerts": 0.0,
            "num_windows": 0
        }

    num_windows = max(1, n // window_size)
    if n >= window_size and n % window_size == 0:
        # Non-overlapping window blocks (for exact block testing)
        blocks = alert_stream.reshape(num_windows, window_size)
        window_sums = np.sum(blocks, axis=1)
    elif n < window_size:
        window_sums = np.array([float(np.sum(alert_stream))])
    else:
        cumsum = np.cumsum(np.insert(alert_stream, 0, 0))
        window_sums = cumsum[window_size:] - cumsum[:-window_size]

    overload_count = np.sum(window_sums > operator_capacity_per_window)
    p_overload = float(overload_count / len(window_sums))
    mean_load = float(np.mean(window_sums))
    peak_load = float(np.max(window_sums))
    total_alerts = float(np.sum(alert_stream))

    return {
        "p_overload": p_overload,
        "overload_probability": p_overload,
        "mean_window_alerts": mean_load,
        "mean_load": mean_load,
        "max_window_alerts": peak_load,
        "peak_load": peak_load,
        "total_alerts": total_alerts,
        "num_windows": num_windows
    }