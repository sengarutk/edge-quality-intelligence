from typing import Dict, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score


def aggregate_anomaly_map(amap: np.ndarray, method: str = "global_max") -> float:
    """
    Applies spatial aggregation rule to reduce a 2D anomaly map to a single scalar image-level score.
    """
    amap_flat = amap.ravel()
    if len(amap_flat) == 0:
        return 0.0

    if method == "global_max":
        return float(np.max(amap_flat))
    elif method == "percentile_99":
        return float(np.percentile(amap_flat, 99.0))
    elif method == "percentile_95":
        return float(np.percentile(amap_flat, 95.0))
    elif method == "top_1_percent_mean":
        k = max(1, int(0.01 * len(amap_flat)))
        top_k = np.partition(amap_flat, -k)[-k:]
        return float(np.mean(top_k))
    elif method == "gaussian_pooled_max":
        blurred = gaussian_filter(amap, sigma=4.0)
        return float(np.max(blurred))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def run_aggregation_ablation(
    pixel_amaps: np.ndarray,
    image_labels: np.ndarray,
    ground_truth_masks: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates metric sensitivity across 5 spatial anomaly map pooling strategies.
    
    Args:
        pixel_amaps: (N, H, W) float anomaly maps.
        image_labels: (N,) binary ground truth labels (0=nominal, 1=defect).
        ground_truth_masks: Optional (N, H, W) binary masks.
        
    Returns:
        Dict mapping strategy name to performance metrics dict.
    """
    strategies = [
        "global_max",
        "percentile_99",
        "percentile_95",
        "top_1_percent_mean",
        "gaussian_pooled_max"
    ]
    
    N = len(pixel_amaps)
    results = {}
    labels = np.asarray(image_labels, dtype=int)
    
    has_two_classes = (len(np.unique(labels)) >= 2)

    for strat in strategies:
        scores = np.zeros(N, dtype=np.float64)
        for i in range(N):
            scores[i] = aggregate_anomaly_map(pixel_amaps[i], method=strat)

        if has_two_classes:
            auroc = float(roc_auc_score(labels, scores))
            ap = float(average_precision_score(labels, scores))
        else:
            auroc = 0.5
            ap = 0.0

        results[strat] = {
            "image_auroc": auroc,
            "image_ap": ap,
            "mean_score_nom": float(np.mean(scores[labels == 0])) if np.sum(labels == 0) > 0 else 0.0,
            "mean_score_def": float(np.mean(scores[labels == 1])) if np.sum(labels == 1) > 0 else 0.0
        }

    return results