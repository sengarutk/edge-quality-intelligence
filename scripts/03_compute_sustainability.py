#!/usr/bin/env python3
"""Quality carbon footprint (QCF) of two image-level thresholds on the MVTec AD score archives.

For every category the confusion counts of two thresholds, both fitted on a calibration half and
applied to the disjoint evaluation half, are annualized with src/sustainability:
  baseline  99th percentile of calibration nominal scores
  CCT       cost-calibrated threshold (r = 10, prior 0.01, at most 5 alerts per 1k on calibration)
This compares thresholds, not the temporal alert policies of the paper.

Part parameters exist only for metal_nut and tile (configs/sustainability_parameters.yaml); grid and
carpet use the tile parameters, and bottle, cable, hazelnut and leather use one generic set of
*assumed* parameters (ASSUMED_PARAMS below). All carbon numbers are therefore illustrative.
Changes are reported with their sign (negative = the CCT threshold increases the footprint).
Outputs: results/benchmark_f1/sustainability/{sustainability_results.tex, sustainability_metrics.tex,
sustainability_summary.json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger

# Anchor project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.cct_ablation import stratified_split_50_50
from src.metrics.cost_calibrated import optimize_cct_threshold
from src.metrics.operational import compute_quantile_threshold
from src.sustainability.qcf_engine import QualityCarbonFootprintEngine, SustainabilityParameters
from src.sustainability.sqi_calculator import SustainabilityQualityIndexCalculator

# Generic part parameters for categories without a measured entry in the YAML file (assumptions).
ASSUMED_PARAMS = dict(m_part=0.50, kappa_mat=2.50, E_rework=2.00, gamma_fatal=0.40, theta_tier=6.0)


def split_counts(files):
    """Confusion counts on evaluation halves for two thresholds fitted on calibration halves.

    baseline: 99th percentile of calibration nominal scores.
    policy:   cost-calibrated threshold (r = 10, prior 0.01, <= 5 alerts per 1k on calibration).
    """
    base = np.zeros(4, dtype=int)  # tp, fp, fn, tn
    pol = np.zeros(4, dtype=int)
    n = 0
    for f in files:
        d = np.load(f)
        seed = int(Path(f).stem.rsplit("_", 1)[-1])
        cs, cy, es, ey = stratified_split_50_50(d["image_labels"], d["image_scores"], seed=seed)
        n += len(ey)
        for tau, acc in ((compute_quantile_threshold(cs[cy == 0], 0.99), base),
                         (optimize_cct_threshold(cs, cy, cost_ratio=10.0, prior=0.01)["threshold"], pol)):
            pred = es >= tau
            acc += [int(np.sum(pred & (ey == 1))), int(np.sum(pred & (ey == 0))),
                    int(np.sum(~pred & (ey == 1))), int(np.sum(~pred & (ey == 0)))]
    return base, pol, n


def compute_sustainability_benchmark() -> Dict[str, Any]:
    """Computes comprehensive QCF and SQI across all 63 benchmark score archives."""
    scores_dir = PROJECT_ROOT / "results" / "benchmark_f1" / "mvtec_ad" / "scores"

    npz_files = sorted(scores_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No score archives found in {scores_dir}")

    logger.info(f"Loaded {len(npz_files)} prediction archives from {scores_dir}")

    config_yaml = PROJECT_ROOT / "configs" / "sustainability_parameters.yaml"
    nut_params = SustainabilityParameters.from_yaml(config_yaml, part_name="metal_nut")
    tile_params = SustainabilityParameters.from_yaml(config_yaml, part_name="tile")

    # Categories to evaluate across all models (PatchCore, PaDiM, AutoEncoder)
    categories = ["bottle", "cable", "carpet", "grid", "hazelnut", "leather", "metal_nut"]
    table_rows = []
    category_metrics = {}

    for cat in categories:
        cat_files = [f for f in npz_files if f.name.startswith(f"{cat}_")]

        if cat == "metal_nut":
            params = nut_params
        elif cat in ["grid", "carpet", "tile"]:
            params = tile_params
        else:
            params = SustainabilityParameters(part_name=cat, **ASSUMED_PARAMS)

        engine = QualityCarbonFootprintEngine(params=params)
        calc = SustainabilityQualityIndexCalculator(qcf_engine=engine)

        (tp_b, fp_b, fn_b, tn_b), (tp_p, fp_p, fn_p, tn_p), n_tot = split_counts(cat_files)

        b_qcf = engine.compute_annual_qcf(tp_b, fp_b, fn_b, tn_b, n_tot)
        p_qcf = engine.compute_annual_qcf(tp_p, fp_p, fn_p, tn_p, n_tot)

        sqi_res = calc.compute_sqi(
            p_qcf, b_qcf,
            {"tp": tp_p, "fp": fp_p, "fn": fn_p, "tn": tn_p},
            {"tp": tp_b, "fp": fp_b, "fn": fn_b, "tn": tn_b},
        )
        msf = sqi_res["msf"]
        esf = sqi_res["esf"]
        csf = sqi_res["csf"]
        cf = sqi_res["cf"]
        sqi = sqi_res["sqi"]

        b_tons = b_qcf["total_qcf_metric_tons"]
        p_tons = p_qcf["total_qcf_metric_tons"]
        red_pct = (1.0 - (p_tons / b_tons)) * 100.0 if b_tons > 1e-6 else 0.0

        table_rows.append({
            "category": cat.replace("_", " ").title(),
            "params": "metal_nut" if cat == "metal_nut" else ("tile" if cat in ("grid", "carpet") else "assumed"),
            "cf": cf,
            "counts_baseline": [int(tp_b), int(fp_b), int(fn_b), int(tn_b)],
            "counts_cct": [int(tp_p), int(fp_p), int(fn_p), int(tn_p)],
            "base_qcf": b_tons,
            "policy_qcf": p_tons,
            "reduction_pct": red_pct,
            "msf": msf,
            "esf": esf,
            "csf": csf,
            "sqi": sqi,
        })
        category_metrics[cat] = {
            "base_qcf": b_tons,
            "policy_qcf": p_tons,
            "reduction_pct": red_pct,
            "msf": msf,
            "esf": esf,
            "sqi": sqi,
        }

    # Dedicated tile parameterization across tile-like classes (grid & carpet)
    tile_engine = QualityCarbonFootprintEngine(params=tile_params)
    tile_calc = SustainabilityQualityIndexCalculator(qcf_engine=tile_engine)
    tile_files = [f for f in npz_files if f.name.startswith("carpet_") or f.name.startswith("grid_")]
    (t_tp_b, t_fp_b, t_fn_b, t_tn_b), (t_tp_p, t_fp_p, t_fn_p, t_tn_p), t_tot = split_counts(tile_files)

    tb_qcf = tile_engine.compute_annual_qcf(t_tp_b, t_fp_b, t_fn_b, t_tn_b, t_tot)
    tp_qcf = tile_engine.compute_annual_qcf(t_tp_p, t_fp_p, t_fn_p, t_tn_p, t_tot)

    tile_sqi_res = tile_calc.compute_sqi(
        tp_qcf, tb_qcf,
        {"tp": t_tp_p, "fp": t_fp_p, "fn": t_fn_p, "tn": t_tn_p},
        {"tp": t_tp_b, "fp": t_fp_b, "fn": t_fn_b, "tn": t_tn_b},
    )
    tile_msf = tile_sqi_res["msf"]
    tile_esf = tile_sqi_res["esf"]
    tile_csf = tile_sqi_res["csf"]
    tile_cf = tile_sqi_res["cf"]
    tile_sqi = tile_sqi_res["sqi"]
    tile_b_tons = tb_qcf["total_qcf_metric_tons"]
    tile_p_tons = tp_qcf["total_qcf_metric_tons"]
    tile_red_pct = (1.0 - (tile_p_tons / tile_b_tons)) * 100.0 if tile_b_tons > 1e-6 else 0.0

    category_metrics["tile"] = {
        "base_qcf": tile_b_tons,
        "policy_qcf": tile_p_tons,
        "reduction_pct": tile_red_pct,
        "msf": tile_msf,
        "esf": tile_esf,
        "sqi": tile_sqi,
    }

    # Generate Booktabs LaTeX Table formatted as single-column float with \resizebox{\columnwidth}{!}
    lines = [
        "\\begin{table}[!b]",
        "\\centering",
        "\\caption{Annual quality carbon footprint of the nominal 99th-percentile threshold (Base) and the cost-calibrated threshold (CCT) on PatchCore, PaDiM and autoencoder scores ($N_{\\text{annual}} = 10^6$ parts). Change: positive = CCT lowers the footprint. Parameters: metal\\_nut and tile from the configuration; grid and carpet use tile; the other categories use assumed generic values (a). Illustrative only.}",
        "\\label{tab:sustainability_results}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Base QCF (t)} & \\textbf{CCT QCF (t)} & \\textbf{Change (\\%)} & \\textbf{MSF} & \\textbf{ESF} & \\textbf{SQI} \\\\",
        "\\midrule",
    ]
    for r in table_rows:
        lines.append(
            f"{r['category']}{' (a)' if r['params'] == 'assumed' else ''} & {r['base_qcf']:.2f} & {r['policy_qcf']:.2f} & {r['reduction_pct']:+.1f}\\% & "
            f"{r['msf']:.3f} & {r['esf']:.3f} & {r['sqi']:.3f} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ])
    table_tex = "\n".join(lines)

    # Write table to both locations
    for target_dir in [PROJECT_ROOT / "results" / "benchmark_f1" / "sustainability"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "sustainability_results.tex").write_text(table_tex, encoding="utf-8")
        logger.info(f"Wrote sustainability table to {target_dir / 'sustainability_results.tex'}")

    # Format macros using \providecommand
    nut_m = category_metrics["metal_nut"]
    tile_m = category_metrics["tile"]
    macros = [
        "",
        "% ESG Sustainability Empirical Macros",
        f"\\providecommand{{\\QCFBaselineMetalNut}}{{{nut_m['base_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFCCTMetalNut}}{{{nut_m['policy_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFChangeMetalNut}}{{{nut_m['reduction_pct']:+.1f}\\%}}",
        f"\\providecommand{{\\SQICCTMetalNut}}{{{nut_m['sqi']:.3f}}}",
        f"\\providecommand{{\\QCFBaselineTile}}{{{tile_m['base_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFCCTTile}}{{{tile_m['policy_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFChangeTile}}{{{tile_m['reduction_pct']:+.1f}\\%}}",
        f"\\providecommand{{\\SQICCTTile}}{{{tile_m['sqi']:.3f}}}",
    ]
    macro_str = "\n".join(macros) + "\n"

    out_dir = PROJECT_ROOT / "results" / "benchmark_f1" / "sustainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    g_file = out_dir / "sustainability_metrics.tex"
    g_file.write_text("% Generated by scripts/03_compute_sustainability.py\n" + macro_str, encoding="utf-8")
    logger.info(f"Wrote QCF macros to {g_file}")

    (out_dir / "sustainability_summary.json").write_text(json.dumps(
        {"compared": ["q99 threshold (baseline)", "CCT threshold"], "assumed_params": ASSUMED_PARAMS,
         "categories": table_rows, "tile_pooled": tile_m}, indent=1), encoding="utf-8")
    return {
        "metal_nut": nut_m,
        "tile": tile_m,
        "table_rows": table_rows,
    }


if __name__ == "__main__":
    compute_sustainability_benchmark()
