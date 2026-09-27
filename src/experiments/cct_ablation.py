import os
import glob
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.metrics.cost_calibrated import optimize_cct_threshold
from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error,
    compute_quantile_threshold,
    compute_alert_budget_threshold
)
from src.metrics.image_metrics import compute_optimal_f1


def stratified_split_50_50(
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into 50% Calibration Set and 50% Evaluation Set with stratified defect proportions.
    Returns: (calib_scores, calib_labels, eval_scores, eval_labels)
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    rng = np.random.RandomState(seed)
    nom_indices = np.where(labels == 0)[0]
    def_indices = np.where(labels == 1)[0]

    rng.shuffle(nom_indices)
    rng.shuffle(def_indices)

    nom_split = len(nom_indices) // 2
    def_split = len(def_indices) // 2

    calib_idx = np.concatenate([nom_indices[:nom_split], def_indices[:def_split]])
    eval_idx = np.concatenate([nom_indices[nom_split:], def_indices[def_split:]])

    rng.shuffle(calib_idx)
    rng.shuffle(eval_idx)

    return (
        scores[calib_idx],
        labels[calib_idx],
        scores[eval_idx],
        labels[eval_idx]
    )


def run_cct_out_of_sample_ablation(
    scores_dir: str = "results/benchmark_f1/mvtec_ad/scores",
    output_dir: str = "results/benchmark_f1/mvtec_ad",
    cost_ratios: List[float] = [10.0, 20.0, 50.0],
    priors: List[float] = [0.01, 0.05, 0.15],
    max_alerts_per_1k: float = 5.0
) -> pd.DataFrame:
    """
    Runs 50/50 out-of-sample calibration vs evaluation across all score archives.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(scores_dir, "*.npz")))
    if len(npz_files) == 0:
        return pd.DataFrame()

    results = []

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

        calib_s, calib_y, eval_s, eval_y = stratified_split_50_50(labels, scores, seed=seed)

        if len(calib_s) == 0 or len(eval_s) == 0:
            continue

        calib_nom = calib_s[calib_y == 0]

        # 1. Nominal 99th Quantile Threshold
        tau_q99 = compute_quantile_threshold(calib_nom, quantile=0.99)
        fa_q99 = compute_fa_at_1k(eval_y, eval_s, tau_q99)
        md_q99 = compute_md_at_1k(eval_y, eval_s, tau_q99)
        cwe_q99_r10 = compute_cost_weighted_error(eval_y, eval_s, tau_q99, cost_ratio=10.0, prior=0.01)
        cwe_q99_r20 = compute_cost_weighted_error(eval_y, eval_s, tau_q99, cost_ratio=20.0, prior=0.01)
        cwe_q99_r50 = compute_cost_weighted_error(eval_y, eval_s, tau_q99, cost_ratio=50.0, prior=0.01)

        # 2. Alert Budget 5 Threshold
        tau_b5 = compute_alert_budget_threshold(calib_nom, max_alerts_per_1k=max_alerts_per_1k)
        fa_b5 = compute_fa_at_1k(eval_y, eval_s, tau_b5)
        md_b5 = compute_md_at_1k(eval_y, eval_s, tau_b5)
        cwe_b5_r10 = compute_cost_weighted_error(eval_y, eval_s, tau_b5, cost_ratio=10.0, prior=0.01)
        cwe_b5_r20 = compute_cost_weighted_error(eval_y, eval_s, tau_b5, cost_ratio=20.0, prior=0.01)
        cwe_b5_r50 = compute_cost_weighted_error(eval_y, eval_s, tau_b5, cost_ratio=50.0, prior=0.01)

        # 3. Cost-Calibrated Threshold (CCT) - Budget-constrained empirical risk minimization
        cct_res_r10 = optimize_cct_threshold(calib_s, calib_y, cost_ratio=10.0, prior=0.01, max_alerts_per_1k=max_alerts_per_1k)
        tau_cct_r10 = cct_res_r10["threshold"]
        # Out-of-sample FA is reported as measured; it can exceed the calibration budget.
        fa_cct_r10 = compute_fa_at_1k(eval_y, eval_s, tau_cct_r10)
        md_cct_r10 = compute_md_at_1k(eval_y, eval_s, tau_cct_r10)
        cwe_cct_r10 = compute_cost_weighted_error(eval_y, eval_s, tau_cct_r10, cost_ratio=10.0, prior=0.01)

        cct_res_r20 = optimize_cct_threshold(calib_s, calib_y, cost_ratio=20.0, prior=0.01, max_alerts_per_1k=max_alerts_per_1k)
        tau_cct_r20 = cct_res_r20["threshold"]
        cwe_cct_r20 = compute_cost_weighted_error(eval_y, eval_s, tau_cct_r20, cost_ratio=20.0, prior=0.01)

        cct_res_r50 = optimize_cct_threshold(calib_s, calib_y, cost_ratio=50.0, prior=0.01, max_alerts_per_1k=max_alerts_per_1k)
        tau_cct_r50 = cct_res_r50["threshold"]
        cwe_cct_r50 = compute_cost_weighted_error(eval_y, eval_s, tau_cct_r50, cost_ratio=50.0, prior=0.01)

        # 4. Oracle F1 on eval set (Theoretical In-Sample Upper Bound)
        oracle_res = compute_optimal_f1(eval_y, eval_s)
        tau_oracle = oracle_res["optimal_threshold"]
        fa_oracle = compute_fa_at_1k(eval_y, eval_s, tau_oracle)
        md_oracle = compute_md_at_1k(eval_y, eval_s, tau_oracle)
        cwe_oracle_r10 = compute_cost_weighted_error(eval_y, eval_s, tau_oracle, cost_ratio=10.0, prior=0.01)

        results.append({
            "category": cat,
            "method": meth,
            "seed": seed,
            # Q99
            "tau_q99": tau_q99,
            "fa_q99": fa_q99,
            "md_q99": md_q99,
            "cwe_q99_r10": cwe_q99_r10,
            "cwe_q99_r20": cwe_q99_r20,
            "cwe_q99_r50": cwe_q99_r50,
            # Budget-5
            "tau_b5": tau_b5,
            "fa_b5": fa_b5,
            "md_b5": md_b5,
            "cwe_b5_r10": cwe_b5_r10,
            "cwe_b5_r20": cwe_b5_r20,
            "cwe_b5_r50": cwe_b5_r50,
            # CCT (Ours)
            "tau_cct_r10": tau_cct_r10,
            "fa_cct_r10": fa_cct_r10,
            "md_cct_r10": md_cct_r10,
            "cwe_cct_r10": cwe_cct_r10,
            "cwe_cct_r20": cwe_cct_r20,
            "cwe_cct_r50": cwe_cct_r50,
            # Oracle
            "tau_oracle": tau_oracle,
            "fa_oracle": fa_oracle,
            "md_oracle": md_oracle,
            "cwe_oracle_r10": cwe_oracle_r10,
            "n_eval_nominal": int(np.sum(eval_y == 0)),
            "fa_resolution_per_1k": 1000.0 / max(1, int(np.sum(eval_y == 0))),
            "budget_violated_out_of_sample": bool(fa_cct_r10 > max_alerts_per_1k),
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(tables_dir, "cct_ablation.csv")
    out_md = os.path.join(tables_dir, "cct_ablation.md")
    out_tex = os.path.join(tables_dir, "cct_ablation.tex")

    df.to_csv(out_csv, index=False)

    summary_df = df.groupby(["category", "method"]).agg({
        "cwe_cct_r10": ["mean", "std"],
        "cwe_b5_r10": ["mean", "std"],
        "cwe_q99_r10": ["mean", "std"],
        "fa_cct_r10": ["mean", "max"],
        "md_cct_r10": "mean",
        "budget_violated_out_of_sample": "mean",
        "fa_resolution_per_1k": "mean",
    }).reset_index()

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Cost-Calibrated Thresholding (CCT) Out-of-Sample Evaluation\n\n")
        f.write(summary_df.to_markdown(index=False))

    # Generate LaTeX table
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\vspace{-2mm}",
        f"\\caption{{Out-of-sample cost-weighted error (CWE, $r=10$, prior $0.01$) with a stratified 50/50 calibration/evaluation split over {len(df)} score archives. FA@1k is measured on the evaluation half; with only about {int(df['n_eval_nominal'].median())} nominal evaluation images its resolution is about {df['fa_resolution_per_1k'].median():.0f} per 1k, so a 5-per-1k budget cannot be verified.}}",
        "\\label{tab:cct_ablation}",
        "\\resizebox{0.95\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{CWE (CCT)} ($\\downarrow$) & \\textbf{CWE (Budget 5)} ($\\downarrow$) & \\textbf{CWE (Quantile 99)} ($\\downarrow$) & \\textbf{FA@1k (CCT, measured)} \\\\",
        "\\midrule"
    ]

    for cat, group in df.groupby("category"):
        cat_display = cat.replace('_', ' ').title()
        for m, m_group in group.groupby("method"):
            m_raw = str(m).lower()
            if "patch" in m_raw:
                m_name = "PatchCore"
            elif "padim" in m_raw:
                m_name = "PaDiM"
            else:
                m_name = "ConvAutoencoder"

            cct_cwe = f"{m_group['cwe_cct_r10'].mean():.4f} \\pm {m_group['cwe_cct_r10'].std():.4f}"
            b5_cwe = f"{m_group['cwe_b5_r10'].mean():.4f} \\pm {m_group['cwe_b5_r10'].std():.4f}"
            q99_cwe = f"{m_group['cwe_q99_r10'].mean():.4f} \\pm {m_group['cwe_q99_r10'].std():.4f}"
            fa_str = f"{m_group['fa_cct_r10'].mean():.1f}"

            lines.append(f"{cat_display} & {m_name} & ${cct_cwe}$ & ${b5_cwe}$ & ${q99_cwe}$ & {fa_str} \\\\")
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

    print(f"✅ CCT Ablation Complete! Saved to {out_csv} and {out_tex}")
    return df
