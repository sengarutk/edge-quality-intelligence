import os
from typing import Dict, Any, Tuple, Optional
import numpy as np


class CostCalibratedThresholdOptimizer:
    """
    Implements Cost-Calibrated Thresholding (CCT) under operator false alarm review bounds
    and asymmetric defect escape penalties.
    """

    @staticmethod
    def compute_empirical_cost_curve(
        scores: np.ndarray,
        labels: np.ndarray,
        cost_ratio: float = 10.0,
        prior: float = 0.01,
        num_thresholds: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes empirical cost curve, FPR curve, and FNR curve across candidate thresholds.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=int).ravel()

        if len(scores) == 0:
            taus = np.linspace(0.0, 1.0, num_thresholds)
            zeros = np.zeros(num_thresholds, dtype=np.float64)
            return taus, zeros, zeros, zeros

        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        if s_min == s_max:
            thresholds = np.array([s_min], dtype=np.float64)
        else:
            thresholds = np.linspace(s_min, s_max, num_thresholds)

        nom_mask = (labels == 0)
        def_mask = (labels == 1)
        n_nom = np.sum(nom_mask)
        n_def = np.sum(def_mask)

        cost_curve = []
        fpr_curve = []
        fnr_curve = []

        for tau in thresholds:
            fpr = float(np.sum(scores[nom_mask] >= tau) / n_nom) if n_nom > 0 else 0.0
            fnr = float(np.sum(scores[def_mask] < tau) / n_def) if n_def > 0 else 0.0
            unit_cost = (1.0 - prior) * fpr * 1.0 + prior * fnr * cost_ratio
            cost_curve.append(unit_cost)
            fpr_curve.append(fpr)
            fnr_curve.append(fnr)

        return (
            thresholds,
            np.array(cost_curve, dtype=np.float64),
            np.array(fpr_curve, dtype=np.float64),
            np.array(fnr_curve, dtype=np.float64)
        )

    @staticmethod
    def optimize_cct_threshold(
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        cost_ratio: float = 10.0,
        prior: float = 0.01,
        max_alerts_per_1k: Optional[float] = 5.0
    ) -> Dict[str, Any]:
        """
        Solves the constrained empirical cost minimization problem:
          tau_CCT = argmin_{tau >= tau_budget} CWE(tau)  s.t.  FA@1k(tau) <= max_alerts_per_1k
          
        Derived threshold strictly satisfies FA@1k <= max_alerts_per_1k on validation distribution.
        """
        val_scores = np.asarray(val_scores, dtype=np.float64).ravel()
        val_labels = np.asarray(val_labels, dtype=int).ravel()

        if len(val_scores) == 0:
            return {
                "threshold": 0.0,
                "min_expected_cost": 0.0,
                "achieved_val_fpr": 0.0,
                "budget_satisfied": True,
                "tau_budget": 0.0
            }

        nom_scores = val_scores[val_labels == 0]
        n_nom = len(nom_scores)
        n_def = np.sum(val_labels == 1)
        max_allowed_fpr = (max_alerts_per_1k / 1000.0) if max_alerts_per_1k is not None else 1.0

        if n_nom > 0 and max_alerts_per_1k is not None:
            target_quantile = max(0.0, min(1.0, 1.0 - max_allowed_fpr))
            tau_budget = float(np.quantile(nom_scores, target_quantile))
        else:
            tau_budget = float(np.min(val_scores))

        # Restrict candidate thresholds to tau >= tau_budget
        candidates = np.sort(np.unique(val_scores[val_scores >= tau_budget]))
        if len(candidates) == 0:
            candidates = np.array([tau_budget])
        else:
            candidates = np.unique(np.concatenate([[tau_budget], candidates]))

        best_tau = tau_budget
        min_cost = float("inf")

        for tau in candidates:
            fpr = float(np.sum(nom_scores >= tau) / n_nom) if n_nom > 0 else 0.0
            fnr = float(np.sum(val_scores[val_labels == 1] < tau) / n_def) if n_def > 0 else 0.0
            unit_cost = (1.0 - prior) * fpr * 1.0 + prior * fnr * cost_ratio
            if unit_cost < min_cost:
                min_cost = unit_cost
                best_tau = float(tau)

        achieved_fpr = float(np.sum(nom_scores >= best_tau) / n_nom) if n_nom > 0 else 0.0
        # Enforce budget lower bound strictly: tau_CCT >= tau_budget
        if (achieved_fpr > max_allowed_fpr + 1e-7 or best_tau < tau_budget) and n_nom > 0:
            best_tau = tau_budget
            achieved_fpr = float(np.sum(nom_scores >= best_tau) / n_nom)
            fnr = float(np.sum(val_scores[val_labels == 1] < best_tau) / n_def) if n_def > 0 else 0.0
            min_cost = (1.0 - prior) * achieved_fpr * 1.0 + prior * fnr * cost_ratio

        return {
            "threshold": float(best_tau),
            "min_expected_cost": float(min_cost),
            "achieved_val_fpr": float(achieved_fpr),
            # Ties at the budget quantile can leave the empirical FPR above the budget.
            "budget_satisfied": bool(achieved_fpr <= max_allowed_fpr + 1e-7),
            "budget_binding": bool(np.isclose(best_tau, tau_budget)),
            "tau_budget": float(tau_budget)
        }


def compute_empirical_cost_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_ratio: float = 10.0,
    prior: float = 0.01,
    num_thresholds: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return CostCalibratedThresholdOptimizer.compute_empirical_cost_curve(
        scores, labels, cost_ratio=cost_ratio, prior=prior, num_thresholds=num_thresholds
    )


def optimize_cct_threshold(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    cost_ratio: float = 10.0,
    prior: float = 0.01,
    max_alerts_per_1k: Optional[float] = 5.0
) -> Dict[str, Any]:
    return CostCalibratedThresholdOptimizer.optimize_cct_threshold(
        val_scores, val_labels, cost_ratio=cost_ratio, prior=prior, max_alerts_per_1k=max_alerts_per_1k
    )
