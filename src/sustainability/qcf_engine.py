"""
Quality Carbon Footprint (QCF) Engine for Edge Quality Intelligence.
Calculates GHG emissions (kg CO2e and metric tons) associated with scrap, rework, escape penalties, and edge compute.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sustainability_parameters.yaml"


@dataclass
class SustainabilityParameters:
    part_name: str = "metal_nut"
    m_part: float = 0.45            # component raw mass (kg)
    kappa_mat: float = 2.89         # material embodied carbon (kg CO2e / kg material)
    E_rework: float = 1.85          # energy required for rework (kWh / part)
    xi_grid: float = 0.230          # grid carbon intensity (kg CO2e / kWh)
    gamma_fatal: float = 0.35       # fraction of defects that are fatal scrap
    gamma_false_scrap: float = 0.05 # fraction of false alarms erroneously scrapped
    theta_tier: float = 8.0         # downstream compounding escape penalty factor
    eta_recycle: float = 0.85       # material circularity / recyclability factor
    P_edge: float = 15.0            # edge runtime power consumption (Watts)
    latency_ms: float = 8.5         # per-frame inference latency (ms); kept for reference only
    N_annual: int = 1_000_000       # parts / year normalization baseline
    operating_hours: float = 6000.0 # hours / year the edge device is powered (assumption, see YAML)
    escape_rework_factor: float = 2.0  # rework energy multiple for a part returned from downstream
    defect_prior: Optional[float] = None  # if set, counts are re-weighted to this production defect rate

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str | Path] = None, part_name: str = "metal_nut") -> "SustainabilityParameters":
        path = Path(yaml_path) if yaml_path else DEFAULT_CONFIG_PATH
        if not path.is_absolute() and not path.exists():
            path = ROOT_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        xi_grid = float(cfg.get("grid_intensity", cfg.get("grid_intensity_xi_grid", 0.230)))
        P_edge = float(cfg.get("edge_power_watts", 15.0))
        N_annual = int(cfg.get("annual_production_volume", 1_000_000))
        operating_hours = float(cfg.get("edge_operating_hours_per_year", 6000.0))
        defect_prior = cfg.get("production_defect_prior")

        parts_cfg = cfg.get("part_types", {})
        if part_name not in parts_cfg:
            raise KeyError(f"Part type '{part_name}' not defined in {path}. Available: {list(parts_cfg.keys())}")

        p = parts_cfg[part_name]
        return cls(
            part_name=part_name,
            m_part=float(p["m_part"]),
            kappa_mat=float(p["kappa_mat"]),
            E_rework=float(p["E_rework"]),
            gamma_fatal=float(p["gamma_fatal"]),
            gamma_false_scrap=float(p.get("gamma_false_scrap", 0.05)),
            theta_tier=float(p["theta_tier"]),
            eta_recycle=float(p.get("eta_recycle", 0.85)),
            xi_grid=xi_grid,
            P_edge=P_edge,
            latency_ms=float(p.get("latency_ms", 8.5)),
            N_annual=N_annual,
            operating_hours=operating_hours,
            defect_prior=None if defect_prior is None else float(defect_prior),
        )


class QualityCarbonFootprintEngine:
    """Annual greenhouse-gas footprint (kg CO2e) of inspection outcomes.

    Per year, with counts scaled to ``annual_production`` parts:
      scrap   = (gamma_fatal * TP + gamma_false_scrap * FP) * m_part * kappa_mat
      rework  = ((1 - gamma_fatal) * TP + (1 - gamma_false_scrap) * FP) * E_rework * xi_grid
      escape  = FN * (theta_tier * m_part * kappa_mat + escape_rework_factor * E_rework * xi_grid)
      compute = P_edge [kW] * operating_hours * xi_grid        (device is powered all year)
    Every flagged part is either scrapped or reworked, never both. MVTec AD test sets are
    defect-heavy; set ``defect_prior`` to re-weight counts to a production defect rate
    (TP/FN are scaled to prior * N, FP/TN to (1 - prior) * N) before annualizing.
    """

    def __init__(
        self,
        params: Optional[SustainabilityParameters] = None,
        m_part: Optional[float] = None,
        kappa_mat: Optional[float] = None,
        E_rework: Optional[float] = None,
        xi_grid: Optional[float] = None,
        gamma_fatal: Optional[float] = None,
        gamma_false_scrap: Optional[float] = None,
        theta_tier: Optional[float] = None,
        P_edge: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ):
        self.params = params or SustainabilityParameters()
        if m_part is not None:
            self.params.m_part = float(m_part)
        if kappa_mat is not None:
            self.params.kappa_mat = float(kappa_mat)
        if E_rework is not None:
            self.params.E_rework = float(E_rework)
        if xi_grid is not None:
            self.params.xi_grid = float(xi_grid)
        if gamma_fatal is not None:
            self.params.gamma_fatal = float(gamma_fatal)
        if gamma_false_scrap is not None:
            self.params.gamma_false_scrap = float(gamma_false_scrap)
        if theta_tier is not None:
            self.params.theta_tier = float(theta_tier)
        if P_edge is not None:
            self.params.P_edge = float(P_edge)
        if latency_ms is not None:
            self.params.latency_ms = float(latency_ms)

    def _scaled_counts(self, tp: float, fp: float, fn: float, tn: float, total: float, annual: float):
        p = self.params
        if total <= 0:
            return 0.0, 0.0, 0.0, 0.0
        if p.defect_prior is None:
            k = annual / total
            return tp * k, fp * k, fn * k, tn * k
        n_def, n_nom = tp + fn, fp + tn
        d, n = p.defect_prior * annual, (1.0 - p.defect_prior) * annual
        tpr = tp / n_def if n_def else 0.0
        fpr = fp / n_nom if n_nom else 0.0
        return tpr * d, fpr * n, (1.0 - tpr) * d, (1.0 - fpr) * n

    def compute_annual_qcf(
        self,
        tp_count: int,
        fp_count: int,
        fn_count: int,
        tn_count: int,
        total_parts: int,
        annual_production: Optional[int] = None,
    ) -> Dict[str, float]:
        p = self.params
        annual = float(annual_production if annual_production is not None else p.N_annual)
        n_tp, n_fp, n_fn, n_tn = self._scaled_counts(tp_count, fp_count, fn_count, tn_count, total_parts, annual)

        scrapped = p.gamma_fatal * n_tp + p.gamma_false_scrap * n_fp
        reworked = (1.0 - p.gamma_fatal) * n_tp + (1.0 - p.gamma_false_scrap) * n_fp
        ghg_scrap = scrapped * p.m_part * p.kappa_mat
        energy_rework = reworked * p.E_rework
        ghg_rework = energy_rework * p.xi_grid
        energy_escape = n_fn * p.escape_rework_factor * p.E_rework
        ghg_escape = n_fn * p.theta_tier * p.m_part * p.kappa_mat + energy_escape * p.xi_grid
        energy_compute = (p.P_edge / 1000.0) * p.operating_hours
        ghg_compute = energy_compute * p.xi_grid

        total = ghg_scrap + ghg_rework + ghg_escape + ghg_compute
        return {
            "ghg_scrap": ghg_scrap,
            "ghg_rework": ghg_rework,
            "ghg_escape": ghg_escape,
            "ghg_compute": ghg_compute,
            "total_qcf_kg": total,
            "total_qcf_metric_tons": total / 1000.0,
            "qcf_annual_kgco2e": total,
            "scrapped_mass_annual_kg": scrapped * p.m_part,
            "energy_annual_kwh": energy_rework + energy_escape + energy_compute,
            "scale_factor": annual / total_parts if total_parts > 0 else 0.0,
        }

    def compute_footprint(
        self,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        latency_sec: float = 0.05,
    ) -> Dict[str, float]:
        n_total = tp + tn + fp + fn
        res = self.compute_annual_qcf(
            tp_count=tp,
            fp_count=fp,
            fn_count=fn,
            tn_count=tn,
            total_parts=n_total,
            annual_production=self.params.N_annual,
        )
        return {
            "n_total": n_total,
            "scrapped_mass_batch_kg": res["scrapped_mass_annual_kg"] * (float(n_total) / float(self.params.N_annual)) if self.params.N_annual > 0 else 0.0,
            "scrapped_mass_annual_kg": res["scrapped_mass_annual_kg"],
            "energy_batch_kwh": res["energy_annual_kwh"] * (float(n_total) / float(self.params.N_annual)) if self.params.N_annual > 0 else 0.0,
            "energy_annual_kwh": res["energy_annual_kwh"],
            "qcf_scrap_kgco2e": res["ghg_scrap"],
            "qcf_rework_kgco2e": res["ghg_rework"],
            "qcf_escape_kgco2e": res["ghg_escape"],
            "qcf_compute_kgco2e": res["ghg_compute"],
            "qcf_batch_kgco2e": res["total_qcf_kg"] * (float(n_total) / float(self.params.N_annual)) if self.params.N_annual > 0 else 0.0,
            "qcf_annual_kgco2e": res["total_qcf_kg"],
            "total_qcf_metric_tons": res["total_qcf_metric_tons"],
        }
