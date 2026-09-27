"""Exact AU-PRO reference, algorithmically independent of ``compute_aupro``.

``compute_aupro`` (pixel_metrics.py) evaluates the PRO curve on a fixed grid of
thresholds. This reference instead sorts all pixels once and accumulates, for every
distinct score, the false-positive rate over normal pixels and the mean per-region
overlap, giving the exact step curve (the approach of the MVTec AD evaluation code).
The curve is integrated up to ``max_fpr`` with linear interpolation at the cut-off and
normalized by ``max_fpr``. Used to validate the grid approximation in tests.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def compute_aupro_reference(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray, max_fpr: float = 0.30,
                            num_thresholds: int | None = None) -> float:
    """Exact AU-PRO. ``num_thresholds`` is accepted for API compatibility and ignored."""
    gt = np.asarray(ground_truth_masks) > 0.5
    scores = np.asarray(anomaly_maps, dtype=np.float64)
    if gt.shape != scores.shape:
        raise ValueError("masks and anomaly maps must have the same shape")

    # Give every connected defect region a global id; 0 marks normal pixels.
    region_ids = np.zeros(gt.shape, dtype=np.int64)
    next_id = 1
    for i in range(gt.shape[0]):
        lab, k = label(gt[i])
        region_ids[i][lab > 0] = lab[lab > 0] + (next_id - 1)
        next_id += k
    n_regions = next_id - 1
    n_normal = int((region_ids == 0).sum())
    if n_regions == 0 or n_normal == 0:
        return 0.0

    ids = region_ids.ravel()
    region_sizes = np.bincount(ids, minlength=n_regions + 1).astype(np.float64)
    order = np.argsort(-scores.ravel(), kind="stable")
    ids_sorted = ids[order]
    s_sorted = scores.ravel()[order]

    fp_step = (ids_sorted == 0).astype(np.float64) / n_normal
    pro_step = np.where(ids_sorted > 0, 1.0 / (n_regions * region_sizes[ids_sorted]), 0.0)
    fpr = np.cumsum(fp_step)
    pro = np.cumsum(pro_step)

    # Keep the last position of each group of tied scores (a threshold admits all ties).
    last_of_group = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    fpr = np.r_[0.0, fpr[last_of_group]]
    pro = np.r_[0.0, pro[last_of_group]]

    keep = fpr <= max_fpr
    x, y = fpr[keep], pro[keep]
    if x[-1] < max_fpr and keep.sum() < fpr.size:
        j = int(keep.sum())  # first point beyond max_fpr
        y_cut = pro[j - 1] + (pro[j] - pro[j - 1]) * (max_fpr - fpr[j - 1]) / (fpr[j] - fpr[j - 1])
        x, y = np.r_[x, max_fpr], np.r_[y, y_cut]
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(y, x) / max_fpr)
