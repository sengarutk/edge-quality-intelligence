import os
import glob
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.metrics.cost_calibrated import optimize_cct_threshold
from src.metrics.operational import compute_quantile_threshold, compute_cost_weighted_error
from src.metrics.image_metrics import compute_image_auroc
from src.experiments.cct_ablation import stratified_split_50_50


def compute_decision_change_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    tau_baseline: float,
    tau_cct: float
) -> Dict[str, Any]:
    """
    Quantifies exact operational decision changes and attribution shifts when switching
    from baseline threshold (tau_baseline) to cost-calibrated threshold (tau_cct).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    n_total = len(scores)

    if n_total == 0:
        return {
            "total_flips": 0,
            "total_flip_rate": 0.0,
            "nominal_relief_count": 0,
            "nominal_relief_rate": 0.0,
            "defect_escape_count": 0,
            "defect_escape_rate": 0.0,
            "defect_catch_count": 0,
            "defect_catch_rate": 0.0,
            "tau_baseline": float(tau_baseline),
            "tau_cct": float(tau_cct)
        }

    pred_base = (scores >= tau_baseline).astype(int)
    pred_cct = (scores >= tau_cct).astype(int)

    nom_mask = (labels == 0)
    def_mask = (labels == 1)
    n_nom = int(np.sum(nom_mask))
    n_def = int(np.sum(def_mask))

    total_flips = int(np.sum(pred_base != pred_cct))
    total_flip_rate = float(total_flips / n_total)

    # False alarms eliminated (Nominal Relief)
    nom_relief = int(np.sum(nom_mask & (pred_base == 1) & (pred_cct == 0)))
    nom_relief_rate = float(nom_relief / n_nom) if n_nom > 0 else 0.0

    # Defects missed due to higher threshold (Defect Escape)
    def_escape = int(np.sum(def_mask & (pred_base == 1) & (pred_cct == 0)))
    def_escape_rate = float(def_escape / n_def) if n_def > 0 else 0.0

    # Defects caught if CCT lowered threshold
    def_catch = int(np.sum(def_mask & (pred_base == 0) & (pred_cct == 1)))
    def_catch_rate = float(def_catch / n_def) if n_def > 0 else 0.0

    return {
        "total_flips": total_flips,
        "total_flip_rate": total_flip_rate,
        "nominal_relief_count": nom_relief,
        "nominal_relief_rate": nom_relief_rate,
        "defect_escape_count": def_escape,
        "defect_escape_rate": def_escape_rate,
        "defect_catch_count": def_catch,
        "defect_catch_rate": def_catch_rate,
        "tau_baseline": float(tau_baseline),
        "tau_cct": float(tau_cct)
    }


def run_decision_change_analysis(
    scores_dir: str = "results/benchmark_f1/mvtec_ad/scores",
    output_dir: str = "results/benchmark_f1/mvtec_ad",
    cost_ratios: List[float] = [5.0, 10.0, 20.0, 50.0]
) -> pd.DataFrame:
    """
    Executes decision change attribution across cost ratios and all benchmark evaluation runs.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(scores_dir, "*.npz")))
    if len(npz_files) == 0:
        return pd.DataFrame()

    records = []

    for fpath in npz_files:
        fname = os.path.basename(fpath).replace(".npz", "")
        parts = fname.split("_")
        if len(parts) >= 3:
            cat, meth, seed = "_".join(parts[:-2]), parts[-2], int(parts[-1])
        else:
            cat, meth, seed = parts[0], parts[1], 42

        data = np.load(fpath)
        labels = data["image_labels"]
        scores = data["image_scores"]

        # Thresholds are fitted on the calibration half and applied to the disjoint evaluation half.
        calib_s, calib_y, scores, labels = stratified_split_50_50(labels, scores, seed=seed)
        tau_99 = compute_quantile_threshold(calib_s[calib_y == 0], quantile=0.99)
        img_auroc = compute_image_auroc(labels, scores)

        for r in cost_ratios:
            cct_res = optimize_cct_threshold(calib_s, calib_y, cost_ratio=r, prior=0.01, max_alerts_per_1k=5.0)
            tau_cct = cct_res["threshold"]
            cwe_val = compute_cost_weighted_error(labels, scores, tau_cct, cost_ratio=r, prior=0.01)

            mat = compute_decision_change_matrix(scores, labels, tau_baseline=tau_99, tau_cct=tau_cct)
            records.append({
                "category": cat,
                "method": meth,
                "seed": seed,
                "cost_ratio": r,
                "image_auroc": img_auroc,
                "cwe": cwe_val,
                "total_flips": mat["total_flips"],
                "total_flip_rate": mat["total_flip_rate"],
                "nominal_relief_count": mat["nominal_relief_count"],
                "nominal_relief_rate": mat["nominal_relief_rate"],
                "defect_escape_count": mat["defect_escape_count"],
                "defect_escape_rate": mat["defect_escape_rate"],
                "defect_catch_count": mat["defect_catch_count"],
                "defect_catch_rate": mat["defect_catch_rate"],
                "tau_99": tau_99,
                "tau_cct": tau_cct
            })

    df = pd.DataFrame(records)
    out_csv = os.path.join(tables_dir, "decision_changes.csv")
    out_md = os.path.join(tables_dir, "decision_changes.md")
    out_tex = os.path.join(tables_dir, "decision_changes.tex")

    df.to_csv(out_csv, index=False)

    summary_df = df.groupby(["category", "method", "cost_ratio"]).agg({
        "total_flips": "mean",
        "total_flip_rate": "mean",
        "nominal_relief_count": "mean",
        "nominal_relief_rate": "mean",
        "defect_escape_count": "mean",
        "defect_escape_rate": "mean",
        "cwe": "mean"
    }).reset_index()

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Decision-Change Attribution Across Defect Escape Cost Ratios\n\n")
        f.write(summary_df.to_markdown(index=False))

    # Compile LaTeX table
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\vspace{-2mm}",
        "\\caption{Decisions that change when the nominal 99th-percentile threshold is replaced by the cost-calibrated threshold (both fitted on the calibration half, applied to the evaluation half). Values are means over seeds; no confidence intervals are reported.}",
        "\\label{tab:decision_changes}",
        "\\resizebox{0.95\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{Cost Ratio ($r$)} & \\textbf{Total Flips} & \\textbf{Nominal Relief (\\%)} $\\uparrow$ & \\textbf{Defect Escape (\\%)} $\\downarrow$ & \\textbf{CWE} $\\downarrow$ \\\\",
        "\\midrule"
    ]

    for cat, group in df.groupby("category"):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{cat.replace('_', ' ').title()}}}}} \\\\")
        for m, m_group in group.groupby("method"):
            m_name = m.replace("patchcore", "PatchCore").replace("padim", "PaDiM").replace("autoencoder", "ConvAutoencoder")
            for r in [10.0, 50.0]:  # Highlight key operational ratios
                sub_r = m_group[m_group["cost_ratio"] == r]
                if len(sub_r) > 0:
                    flips = f"{sub_r['total_flips'].mean():.1f}"
                    relief = f"{sub_r['nominal_relief_rate'].mean() * 100:.1f}\\%"
                    escape = f"{sub_r['defect_escape_rate'].mean() * 100:.1f}\\%"
                    cwe_str = f"{sub_r['cwe'].mean():.4f}"
                    lines.append(f" & {m_name} & {int(r)}$\\times$ & {flips} & {relief} & {escape} & {cwe_str} \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}"
    ])

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✅ Decision-Change Attribution Complete! Saved to {out_csv} and {out_tex}")
    return df