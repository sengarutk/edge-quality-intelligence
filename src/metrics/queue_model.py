"""Queueing models of the operator review station.

Arrivals are operator alerts, served by one reviewer at rate ``mu`` (reviews per hour).
The analytical models assume Poisson arrivals; the simulation keeps Poisson arrivals but
allows log-normal review times. Alerts produced by a temporal policy are burstier than
Poisson, so the analytical numbers are a lower bound on the real waiting time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class OperatorQueueModel:
    """M/M/1, M/D/1 and M/G/1 (log-normal service, simulated) review-queue models."""

    def __init__(self, service_rate_per_hour: float = 60.0) -> None:
        if service_rate_per_hour <= 0:
            raise ValueError("service_rate_per_hour must be positive")
        self.mu = float(service_rate_per_hour)

    def _unstable(self, rho: float) -> Dict[str, Any]:
        return {"utilization": rho, "stable": False, "mean_queue_length": float("inf"),
                "mean_wait_time_minutes": float("inf"), "p_at_least_10_in_system": None}

    def analyze_mm1(self, arrival_rate_per_hour: float) -> Dict[str, Any]:
        """Steady-state M/M/1: L_q = rho^2/(1-rho), W_q = rho/(mu(1-rho)), P(N >= 10) = rho^10."""
        lam = float(arrival_rate_per_hour)
        if lam < 0:
            raise ValueError("arrival rate must be non-negative")
        rho = lam / self.mu
        if rho >= 1.0:
            return self._unstable(rho)
        return {
            "utilization": rho,
            "stable": True,
            "mean_queue_length": rho ** 2 / (1.0 - rho),
            "mean_wait_time_minutes": 60.0 * rho / (self.mu * (1.0 - rho)),
            "p_at_least_10_in_system": rho ** 10,
        }

    def analyze_md1(self, arrival_rate_per_hour: float) -> Dict[str, Any]:
        """Steady-state M/D/1 (Pollaczek-Khinchine with zero service variance)."""
        lam = float(arrival_rate_per_hour)
        if lam < 0:
            raise ValueError("arrival rate must be non-negative")
        rho = lam / self.mu
        if rho >= 1.0:
            return self._unstable(rho)
        return {
            "utilization": rho,
            "stable": True,
            "mean_queue_length": rho ** 2 / (2.0 * (1.0 - rho)),
            "mean_wait_time_minutes": 60.0 * rho / (2.0 * self.mu * (1.0 - rho)),
            "p_at_least_10_in_system": None,  # no simple closed form for M/D/1
        }

    def simulate_variable_service(
        self,
        arrival_rate_per_hour: float,
        duration_hours: float = 8.0,
        service_sigma: float = 0.35,
        seed: int = 42,
        arrival_times_min: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Single-server FIFO simulation with log-normal service times (mean 60/mu minutes).

        ``arrival_times_min`` may be supplied to replay a recorded alert stream instead of
        Poisson arrivals. The time-averaged queue length L_q is computed exactly from the
        piecewise-constant queue trajectory over the horizon; ``backlog_at_end`` exposes
        instability (it grows with the horizon when utilization >= 1).
        """
        horizon = duration_hours * 60.0
        rng = np.random.RandomState(seed)
        if arrival_times_min is None:
            n = rng.poisson(max(0.0, arrival_rate_per_hour) * duration_hours)
            arrivals = np.sort(rng.uniform(0.0, horizon, size=n))
        else:
            arrivals = np.sort(np.asarray(arrival_times_min, dtype=np.float64))
        n = arrivals.size
        empty = {"mean_queue_length": 0.0, "mean_wait_minutes": 0.0, "p95_wait_minutes": 0.0,
                 "max_wait_minutes": 0.0, "backlog_at_end": 0, "n_arrivals": 0}
        if n == 0:
            return empty

        mean_service = 60.0 / self.mu
        mu_log = np.log(mean_service) - 0.5 * service_sigma ** 2
        service = rng.lognormal(mean=mu_log, sigma=service_sigma, size=n)

        start = np.empty(n)
        free_at = 0.0
        for i in range(n):
            start[i] = max(arrivals[i], free_at)
            free_at = start[i] + service[i]
        waits = start - arrivals

        # Queue length (excluding the item in service) is +1 at each arrival and -1 at each
        # service start; integrate it over [0, horizon].
        events = np.concatenate([np.c_[arrivals, np.ones(n)], np.c_[start, -np.ones(n)]])
        events = events[np.lexsort((events[:, 1], events[:, 0]))]
        t_prev, q, area = 0.0, 0, 0.0
        for t, d in events:
            t_clip = min(t, horizon)
            area += q * max(0.0, t_clip - t_prev)
            t_prev = max(t_prev, t_clip)
            q += int(d)
        area += q * max(0.0, horizon - t_prev)
        backlog_at_end = int(np.sum(arrivals <= horizon) - np.sum(start <= horizon))

        return {
            "mean_queue_length": float(area / horizon),
            "mean_wait_minutes": float(np.mean(waits)),
            "p95_wait_minutes": float(np.percentile(waits, 95)),
            "max_wait_minutes": float(np.max(waits)),
            "backlog_at_end": backlog_at_end,
            "n_arrivals": int(n),
        }

    def sweep_service_variability(
        self,
        arrival_rates: List[float],
        sigmas: Optional[List[float]] = None,
        duration_hours: float = 8.0,
        n_trials: int = 20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Mean over ``n_trials`` simulations for each (sigma, arrival rate)."""
        sigmas = sigmas if sigmas is not None else [0.2, 0.4, 0.6]
        results: Dict[str, Any] = {}
        for sigma in sigmas:
            key = f"sigma_{sigma:.1f}"
            results[key] = {"arrival_rates": list(arrival_rates), "mean_queue_lengths": [],
                            "mean_wait_times_min": [], "p95_wait_times_min": []}
            for lam in arrival_rates:
                sims = [self.simulate_variable_service(lam, duration_hours, sigma, seed + 1000 * t)
                        for t in range(n_trials)]
                results[key]["mean_queue_lengths"].append(float(np.mean([s["mean_queue_length"] for s in sims])))
                results[key]["mean_wait_times_min"].append(float(np.mean([s["mean_wait_minutes"] for s in sims])))
                results[key]["p95_wait_times_min"].append(float(np.mean([s["p95_wait_minutes"] for s in sims])))
        return results
