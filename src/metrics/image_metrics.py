import warnings
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        warnings.warn("compute_image_auroc: Labels contain only one class. Returning 0.5.")
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_image_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        warnings.warn("compute_image_ap: Labels contain only one class. Returning 0.0.")
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_quantile_threshold(nominal_scores: np.ndarray, quantile: float = 0.99) -> float:
    if len(nominal_scores) == 0:
        return 0.5
    return float(np.percentile(nominal_scores, quantile * 100.0))


def compute_optimal_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        med = float(np.mean(scores)) if len(scores) > 0 else 0.5
        return {
            "max_f1": 0.0,
            "optimal_threshold": med,
            "oracle_max_f1": 0.0,
            "oracle_threshold": med,
            "precision": 0.0,
            "recall": 0.0,
            "precision_at_optimal": 0.0,
            "recall_at_optimal": 0.0
        }

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    f1_scores = np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        p = precisions[i]
        r = recalls[i]
        if (p + r) > 0:
            f1_scores[i] = (2 * p * r) / (p + r)
        else:
            f1_scores[i] = 0.0

    if len(f1_scores) == 0 or np.max(f1_scores) == 0:
        med = float(np.median(scores))
        return {
            "max_f1": 0.0,
            "optimal_threshold": med,
            "oracle_max_f1": 0.0,
            "oracle_threshold": med,
            "precision": 0.0,
            "recall": 0.0,
            "precision_at_optimal": 0.0,
            "recall_at_optimal": 0.0
        }

    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_th = float(thresholds[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    return {
        "max_f1": best_f1,
        "optimal_threshold": best_th,
        "oracle_max_f1": best_f1,
        "oracle_threshold": best_th,
        "precision": best_prec,
        "recall": best_rec,
        "precision_at_optimal": best_prec,
        "recall_at_optimal": best_rec
    }


compute_optimal_f1_threshold = compute_optimal_f1


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return compute_image_auroc(labels, scores)