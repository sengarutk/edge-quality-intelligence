"""
Sustainable Quality Index (SQI) Calculator.
Combines Material Savings Factor (MSF), Energy Savings Factor (ESF),
Carbon Savings Factor (CSF), and Circularity Factor (CF) into a unified [0.0, 1.0] metric.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from src.sustainability.qcf_engine import QualityCarbonFootprintEngine, SustainabilityParameters


class SustainabilityQualityIndexCalculator:
    """
    Computes Sustainable Quality Index (SQI in [0.0, 1.0]).
    SQI = 0.30 * MSF + 0.25 * ESF + 0.30 * CSF + 0.15 * CF
    """
    def __init__(
        self,
        qcf_engine: Optional[QualityCarbonFootprintEngine] = None,
        params: Optional[SustainabilityParameters] = None,
    ):
        self.qcf_engine = qcf_engine or QualityCarbonFootprintEngine(params=params)
        self.params = self.qcf_engine.params

    def compute_sqi(
        self,
        policy_qcf: Dict[str, float],
        baseline_qcf: Dict[str, float],
        policy_counts: Dict[str, int],
        baseline_counts: Dict[str, int],
    ) -> Dict[str, float]:
        # Scrap Mass
        m_base = baseline_qcf.get("scrapped_mass_annual_kg", 0.0)
        m_pol = policy_qcf.get("scrapped_mass_annual_kg", 0.0)
        if m_base > 1e-9:
            msf = max(0.0, min(1.0, 1.0 - (m_pol / m_base)))
        else:
            msf = 1.0 if m_pol <= 1e-9 else 0.0

        # Energy
        e_base = baseline_qcf.get("energy_annual_kwh", 0.0)
        e_pol = policy_qcf.get("energy_annual_kwh", 0.0)
        if e_base > 1e-9:
            esf = max(0.0, min(1.0, 1.0 - (e_pol / e_base)))
        else:
            esf = 1.0 if e_pol <= 1e-9 else 0.0

        # Carbon / QCF
        c_base = baseline_qcf.get("total_qcf_kg", baseline_qcf.get("qcf_annual_kgco2e", 0.0))
        c_pol = policy_qcf.get("total_qcf_kg", policy_qcf.get("qcf_annual_kgco2e", 0.0))
        if c_base > 1e-9:
            csf = max(0.0, min(1.0, 1.0 - (c_pol / c_base)))
        else:
            csf = 1.0 if c_pol <= 1e-9 else 0.0

        # Circularity factor: share of parts that stay in use (TN, reworked TP, reworked FP)
        # plus the recycled share (eta_recycle) of scrapped TP and FP.
        p_tp = float(policy_counts.get("tp", 0))
        p_tn = float(policy_counts.get("tn", 0))
        p_fp = float(policy_counts.get("fp", 0))
        p_fn = float(policy_counts.get("fn", 0))
        n_total = p_tp + p_tn + p_fp + p_fn

        gamma_fatal = self.params.gamma_fatal
        if n_total > 0:
            g_fs = self.params.gamma_false_scrap
            eta = self.params.eta_recycle
            kept = p_tn + (1.0 - gamma_fatal) * p_tp + (1.0 - g_fs) * p_fp
            recycled = eta * (gamma_fatal * p_tp + g_fs * p_fp)
            cf = (kept + recycled) / float(n_total)
            cf = max(0.0, min(1.0, cf))
        else:
            cf = 1.0

        # SQI = 0.30 * MSF + 0.25 * ESF + 0.30 * CSF + 0.15 * CF
        raw_sqi = 0.30 * msf + 0.25 * esf + 0.30 * csf + 0.15 * cf
        sqi = max(0.0, min(1.0, raw_sqi))

        return {
            "msf": msf,
            "esf": esf,
            "csf": csf,
            "cf": cf,
            "sqi": sqi,
            "policy_qcf_annual": c_pol,
            "baseline_qcf_annual": c_base,
            "policy_energy_annual": e_pol,
            "baseline_energy_annual": e_base,
            "policy_mass_annual": m_pol,
            "baseline_mass_annual": m_base,
        }

    def calculate_sqi(
        self,
        policy_counts: Dict[str, int],
        baseline_counts: Dict[str, int],
        policy_latency: float = 0.05,
        baseline_latency: float = 0.05,
    ) -> Dict[str, float]:
        pol_qcf = self.qcf_engine.compute_footprint(
            tp=policy_counts.get("tp", 0),
            tn=policy_counts.get("tn", 0),
            fp=policy_counts.get("fp", 0),
            fn=policy_counts.get("fn", 0),
            latency_sec=policy_latency,
        )
        base_qcf = self.qcf_engine.compute_footprint(
            tp=baseline_counts.get("tp", 0),
            tn=baseline_counts.get("tn", 0),
            fp=baseline_counts.get("fp", 0),
            fn=baseline_counts.get("fn", 0),
            latency_sec=baseline_latency,
        )
        return self.compute_sqi(
            policy_qcf=pol_qcf,
            baseline_qcf=base_qcf,
            policy_counts=policy_counts,
            baseline_counts=baseline_counts,
        )


SQICalculator = SustainabilityQualityIndexCalculator
