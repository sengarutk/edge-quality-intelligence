from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error,
    compute_quantile_threshold,
    compute_alert_budget_threshold,
    compute_validation_cost_optimal_threshold,
    compute_tpr_at_alert_budget,
    compute_operator_overload
)
from src.metrics.image_metrics import compute_optimal_f1


class ProductionStreamSimulator:
    """
    Simulates high-throughput production inspection streams under realistic factory operational regimes:
      1. IID Stream: Standard independent resampling from nominal and defect score distributions.
      2. Block-Correlated (Burst) Stream: Two-state Markov chain modeling intermittent tooling/batch failure bursts.
      3. Gradual Drift Stream: Simulates sensor thermal drift, lens fouling, or progressive illumination decay.
    """
    def __init__(self, nominal_scores: np.ndarray, defect_scores: np.ndarray, seed: int = 42):
        self.nominal_scores = np.asarray(nominal_scores, dtype=np.float64).ravel()
        self.defect_scores = np.asarray(defect_scores, dtype=np.float64).ravel()
        self.rng = np.random.RandomState(seed)

    def simulate_iid_stream(
        self,
        n_total: int = 10000,
        defect_prior: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates an IID production sequence with exact target defect occurrences.
        """
        if len(self.nominal_scores) == 0 and len(self.defect_scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        if len(self.defect_scores) == 0:
            labels = np.zeros(n_total, dtype=int)
            scores = self.rng.choice(self.nominal_scores, size=n_total, replace=True)
            return labels, scores

        if len(self.nominal_scores) == 0:
            labels = np.ones(n_total, dtype=int)
            scores = self.rng.choice(self.defect_scores, size=n_total, replace=True)
            return labels, scores

        def_count = int(round(n_total * defect_prior))
        nom_count = n_total - def_count

        labels = np.concatenate([np.zeros(nom_count, dtype=int), np.ones(def_count, dtype=int)])
        perm = self.rng.permutation(n_total)
        labels = labels[perm]

        scores = np.zeros(n_total, dtype=np.float64)
        if nom_count > 0:
            scores[labels == 0] = self.rng.choice(self.nominal_scores, size=nom_count, replace=True)
        if def_count > 0:
            scores[labels == 1] = self.rng.choice(self.defect_scores, size=def_count, replace=True)

        return labels, scores

    def simulate_block_correlated_stream(
        self,
        n_total: int = 10000,
        defect_prior: float = 0.01,
        mean_block_length: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates factory burst/tooling defects using a Two-State Markov Chain.
          State 0 (Nominal): P(0 -> 1) = defect_prior / (mean_block_length * (1 - defect_prior))
          State 1 (Defective Burst): P(1 -> 1) = 1 - (1 / mean_block_length)
        """
        if len(self.nominal_scores) == 0 or len(self.defect_scores) == 0:
            return self.simulate_iid_stream(n_total=n_total, defect_prior=defect_prior)

        mean_L = max(1.0, float(mean_block_length))
        p11 = 1.0 - (1.0 / mean_L)
        p01 = (defect_prior / (mean_L * max(1e-5, (1.0 - defect_prior))))
        p01 = max(0.0, min(1.0, p01))

        states = np.zeros(n_total, dtype=int)
        curr_state = 1 if self.rng.rand() < defect_prior else 0
        states[0] = curr_state

        for t in range(1, n_total):
            r = self.rng.rand()
            if curr_state == 0:
                if r < p01:
                    curr_state = 1
            else:
                if r > p11:
                    curr_state = 0
            states[t] = curr_state

        scores = np.zeros(n_total, dtype=np.float64)
        nom_count = int(np.sum(states == 0))
        def_count = int(np.sum(states == 1))

        if nom_count > 0:
            scores[states == 0] = self.rng.choice(self.nominal_scores, size=nom_count, replace=True)
        if def_count > 0:
            scores[states == 1] = self.rng.choice(self.defect_scores, size=def_count, replace=True)

        return states, scores

    def simulate_drift_stream(
        self,
        n_total: int = 10000,
        defect_prior: float = 0.01,
        drift_slope: float = 0.10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates gradual optical fouling / illumination drift over an inspection shift.
        """
        labels, scores = self.simulate_iid_stream(n_total=n_total, defect_prior=defect_prior)
        if len(scores) == 0:
            return labels, scores

        sigma_nom = float(np.std(self.nominal_scores)) if len(self.nominal_scores) > 1 else 1.0
        if sigma_nom == 0:
            sigma_nom = 1.0

        t_steps = np.linspace(0.0, 1.0, n_total)
        drift_additive = t_steps * drift_slope * sigma_nom
        drifted_scores = scores + drift_additive
        return labels, drifted_scores

    def simulate_stream(
        self,
        n_total: int = 10000,
        defect_prior: float = 0.01,
        regime: str = "iid",
        mean_block_length: int = 20,
        drift_slope: float = 0.10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unified simulator dispatching to the chosen operational regime.
        """
        if regime == "burst":
            return self.simulate_block_correlated_stream(
                n_total=n_total,
                defect_prior=defect_prior,
                mean_block_length=mean_block_length
            )
        elif regime == "drift":
            return self.simulate_drift_stream(
                n_total=n_total,
                defect_prior=defect_prior,
                drift_slope=drift_slope
            )
        else:
            return self.simulate_iid_stream(n_total=n_total, defect_prior=defect_prior)

    def evaluate_stream_robustness(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        tau: float,
        cost_ratio: float = 10.0,
        window_size: int = 1000,
        operator_capacity: int = 60
    ) -> Dict[str, float]:
        """
        Evaluates operational performance and operator overload on an instantiated stream.
        """
        if len(labels) == 0:
            return {
                "fa_at_1k": 0.0,
                "md_at_1k": 0.0,
                "cwe": 0.0,
                "p_overload": 0.0,
                "overload_probability": 0.0,
                "mean_window_alerts": 0.0,
                "mean_load": 0.0,
                "max_window_alerts": 0.0,
                "peak_load": 0.0,
                "total_alerts": 0.0
            }

        fa_1k = compute_fa_at_1k(labels, scores, tau)
        md_1k = compute_md_at_1k(labels, scores, tau)
        cwe = compute_cost_weighted_error(labels, scores, tau, cost_ratio=cost_ratio)
        alerts = (scores >= tau).astype(int)
        overload = compute_operator_overload(
            alerts,
            operator_capacity_per_window=operator_capacity,
            window_size=window_size
        )

        return {
            "fa_at_1k": fa_1k,
            "md_at_1k": md_1k,
            "cwe": cwe,
            "p_overload": overload["p_overload"],
            "overload_probability": overload["overload_probability"],
            "mean_window_alerts": overload["mean_window_alerts"],
            "mean_load": overload["mean_load"],
            "max_window_alerts": overload["max_window_alerts"],
            "peak_load": overload["peak_load"],
            "total_alerts": overload["total_alerts"]
        }

    def evaluate_threshold_strategies(
        self,
        stream_labels: np.ndarray,
        stream_scores: np.ndarray,
        nominal_ref_scores: Optional[np.ndarray] = None,
        cost_ratio: float = 10.0,
        defect_prior: float = 0.01
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates oracle F1, nominal 99th quantile, and cost-optimal strategies on a simulated stream.
        """
        if nominal_ref_scores is None or len(nominal_ref_scores) == 0:
            nominal_ref_scores = self.nominal_scores if len(self.nominal_scores) > 0 else stream_scores[stream_labels == 0]

        # 1. Oracle Max F1 threshold (evaluated on stream labels & scores)
        f1_res = compute_optimal_f1(stream_labels, stream_scores)
        tau_oracle = f1_res["optimal_threshold"]
        rob_oracle = self.evaluate_stream_robustness(stream_labels, stream_scores, tau_oracle, cost_ratio=cost_ratio)
        res_oracle = {
            "threshold": tau_oracle,
            "fa_at_1k": rob_oracle["fa_at_1k"],
            "md_at_1k": rob_oracle["md_at_1k"],
            "cwe": rob_oracle["cwe"],
            "tpr": 1.0 - (rob_oracle["md_at_1k"] / 1000.0)
        }

        # 2. Nominal Quantile 99% threshold
        tau_q99 = compute_quantile_threshold(nominal_ref_scores, quantile=0.99)
        rob_q99 = self.evaluate_stream_robustness(stream_labels, stream_scores, tau_q99, cost_ratio=cost_ratio)
        res_q99 = {
            "threshold": tau_q99,
            "fa_at_1k": rob_q99["fa_at_1k"],
            "md_at_1k": rob_q99["md_at_1k"],
            "cwe": rob_q99["cwe"],
            "tpr": 1.0 - (rob_q99["md_at_1k"] / 1000.0)
        }

        # 3. Cost-Optimal Threshold
        nom_pool = self.nominal_scores if len(self.nominal_scores) > 0 else stream_scores[stream_labels == 0]
        def_pool = self.defect_scores if len(self.defect_scores) > 0 else stream_scores[stream_labels == 1]
        tau_cost = compute_validation_cost_optimal_threshold(
            nom_pool, def_pool, cost_ratio=cost_ratio, prior=defect_prior
        )
        rob_cost = self.evaluate_stream_robustness(stream_labels, stream_scores, tau_cost, cost_ratio=cost_ratio)
        res_cost = {
            "threshold": tau_cost,
            "fa_at_1k": rob_cost["fa_at_1k"],
            "md_at_1k": rob_cost["md_at_1k"],
            "cwe": rob_cost["cwe"],
            "tpr": 1.0 - (rob_cost["md_at_1k"] / 1000.0)
        }

        return {
            "oracle_f1": res_oracle,
            "nominal_quantile_99": res_q99,
            "cost_optimal": res_cost
        }


def evaluate_threshold_strategies(
    test_nominal_scores: np.ndarray,
    test_defect_scores: np.ndarray,
    ref_nominal_scores: Optional[np.ndarray] = None,
    val_nominal_scores: Optional[np.ndarray] = None,
    val_defect_scores: Optional[np.ndarray] = None,
    cost_ratios: List[float] = [10.0, 20.0, 50.0],
    priors: List[float] = [0.01, 0.05, 0.15],
    alert_budgets: List[float] = [5.0, 10.0],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Evaluates deployable, leakage-free threshold strategies against oracle baselines.
    """
    if ref_nominal_scores is None or len(ref_nominal_scores) == 0:
        raise ValueError("ref_nominal_scores must come from data disjoint from the test pools (no test-label leakage)")

    results = []
    simulator = ProductionStreamSimulator(test_nominal_scores, test_defect_scores, seed=seed)

    for budget in alert_budgets:
        tau_budget = compute_alert_budget_threshold(ref_nominal_scores, max_alerts_per_1k=budget)
        for prior in priors:
            labels_iid, scores_iid = simulator.simulate_iid_stream(10000, defect_prior=prior)
            labels_burst, scores_burst = simulator.simulate_block_correlated_stream(10000, defect_prior=prior)
            labels_drift, scores_drift = simulator.simulate_drift_stream(10000, defect_prior=prior)

            for cost_r in cost_ratios:
                res_iid = simulator.evaluate_stream_robustness(labels_iid, scores_iid, tau_budget, cost_ratio=cost_r)
                res_burst = simulator.evaluate_stream_robustness(labels_burst, scores_burst, tau_budget, cost_ratio=cost_r)
                res_drift = simulator.evaluate_stream_robustness(labels_drift, scores_drift, tau_budget, cost_ratio=cost_r)

                results.append({
                    "strategy": f"alert_budget_{int(budget)}",
                    "threshold_type": "alert_budget",
                    "threshold_value": tau_budget,
                    "budget_per_1k": budget,
                    "prior": prior,
                    "cost_ratio": cost_r,
                    "fa_at_1k": res_iid["fa_at_1k"],
                    "md_at_1k": res_iid["md_at_1k"],
                    "cwe": res_iid["cwe"],
                    "p_overload_iid": res_iid["p_overload"],
                    "p_overload_burst": res_burst["p_overload"],
                    "p_overload_drift": res_drift["p_overload"],
                    "tpr": 1.0 - (res_iid["md_at_1k"] / 1000.0)
                })

    tau_q99 = compute_quantile_threshold(ref_nominal_scores, quantile=0.99)
    for prior in priors:
        labels_iid, scores_iid = simulator.simulate_iid_stream(10000, defect_prior=prior)
        for cost_r in cost_ratios:
            res_iid = simulator.evaluate_stream_robustness(labels_iid, scores_iid, tau_q99, cost_ratio=cost_r)
            results.append({
                "strategy": "quantile_99_nominal",
                "threshold_type": "quantile_99",
                "threshold_value": tau_q99,
                "budget_per_1k": None,
                "prior": prior,
                "cost_ratio": cost_r,
                "fa_at_1k": res_iid["fa_at_1k"],
                "md_at_1k": res_iid["md_at_1k"],
                "cwe": res_iid["cwe"],
                "p_overload_iid": res_iid["p_overload"],
                "p_overload_burst": None,
                "p_overload_drift": None,
                "tpr": 1.0 - (res_iid["md_at_1k"] / 1000.0)
            })

    return results


class MixedCorruptionStreamSimulator:
    """
    Simulates factory inspection stream with stochastic environmental degradation:
    Each incoming item has probability p_corr of suffering physical corruptions
    (defocus blur, lighting shift, sensor noise).
    """
    def __init__(
        self,
        nominal_clean_scores: np.ndarray,
        defect_clean_scores: np.ndarray,
        nominal_corr_scores: np.ndarray,
        defect_corr_scores: np.ndarray,
        seed: int = 42
    ):
        self.nominal_clean = np.asarray(nominal_clean_scores, dtype=np.float64).ravel()
        self.defect_clean = np.asarray(defect_clean_scores, dtype=np.float64).ravel()
        self.nominal_corr = np.asarray(nominal_corr_scores, dtype=np.float64).ravel()
        self.defect_corr = np.asarray(defect_corr_scores, dtype=np.float64).ravel()
        self.rng = np.random.RandomState(seed)

    def simulate_mixed_stream(
        self,
        n_total: int = 10000,
        defect_prior: float = 0.01,
        corruption_prob: float = 0.20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates stream returning (labels, scores, is_corrupted).
        """
        if len(self.nominal_clean) == 0 and len(self.defect_clean) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64), np.array([], dtype=bool)

        def_count = int(round(n_total * defect_prior))
        nom_count = n_total - def_count

        labels = np.concatenate([np.zeros(nom_count, dtype=int), np.ones(def_count, dtype=int)])
        perm = self.rng.permutation(n_total)
        labels = labels[perm]

        is_corrupted = self.rng.rand(n_total) < corruption_prob
        scores = np.zeros(n_total, dtype=np.float64)

        for i in range(n_total):
            y = labels[i]
            corr = is_corrupted[i]
            if y == 0:
                pool = self.nominal_corr if (corr and len(self.nominal_corr) > 0) else self.nominal_clean
            else:
                pool = self.defect_corr if (corr and len(self.defect_corr) > 0) else self.defect_clean

            if len(pool) > 0:
                scores[i] = self.rng.choice(pool)
            else:
                scores[i] = 0.0

        return labels, scores, is_corrupted

    def evaluate_mixed_stream(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        tau: float,
        cost_ratio: float = 10.0,
        prior: float = 0.01
    ) -> Dict[str, float]:
        fa = compute_fa_at_1k(labels, scores, tau)
        md = compute_md_at_1k(labels, scores, tau)
        cwe = compute_cost_weighted_error(labels, scores, tau, cost_ratio=cost_ratio)
        return {
            "fa_at_1k": float(fa),
            "md_at_1k": float(md),
            "cwe": float(cwe)
        }