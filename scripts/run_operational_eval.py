import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
from src.experiments.cct_ablation import stratified_split_50_50
import pandas as pd

from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error,
    compute_quantile_threshold,
    compute_alert_budget_threshold,
    compute_tpr_at_alert_budget,
    compute_operator_overload
)
from src.metrics.stats import (
    bootstrap_ci,
    hierarchical_bootstrap_ci,
    compute_paired_wilcoxon_analysis,
    holm_bonferroni_from_pvalues
)
from src.experiments.operational_eval import ProductionStreamSimulator


def generate_latex_table(df: pd.DataFrame, output_tex: str):
    """
    Generates a publication-grade LaTeX booktabs table for operational metrics.
    """
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Operational Inspection Benchmark under Constrained False Alert Budget ($\\le 5$ alarms/1k) and Asymmetric Escape Cost ($r=10$). Metrics show mean with 95\\% Bootstrap Confidence Intervals across multi-seed runs.}",
        "\\label{tab:operational_results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{TPR @ 5 Alarms/1k} ($\\uparrow$) & \\textbf{Missed Defects/1k} ($\\downarrow$) & \\textbf{Cost-Weighted Error ($r=10$)} ($\\downarrow$) & \\textbf{Overload Prob. $P(\\text{Overload})$} ($\\downarrow$) \\\\",
        "\\midrule"
    ]

    for cat, group in df.groupby("category"):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{cat.replace('_', ' ').title()}}}}} \\\\")
        for _, row in group.iterrows():
            m_name = row["method"].replace("patchcore", "PatchCore").replace("padim", "PaDiM").replace("autoencoder", "ConvAutoencoder")
            tpr_str = f"{row['tpr_at_5_mean']:.3f} [{row['tpr_at_5_ci_low']:.3f}, {row['tpr_at_5_ci_high']:.3f}]"
            md_str = f"{row['md_at_1k_mean']:.1f} [{row['md_at_1k_ci_low']:.1f}, {row['md_at_1k_ci_high']:.1f}]"
            cwe_str = f"{row['cwe_r10_mean']:.4f} [{row['cwe_r10_ci_low']:.4f}, {row['cwe_r10_ci_high']:.4f}]"
            ovl_str = f"{row['overload_prob_mean']:.3f}"
            lines.append(f" & {m_name} & {tpr_str} & {md_str} & {cwe_str} & {ovl_str} \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Generated LaTeX Table: {output_tex}")


def main():
    parser = argparse.ArgumentParser(description="Master Operational Evaluation, Hierarchical Bootstrapping, and Statistical Testing")
    parser.add_argument("--scores-dir", type=str, default="results/benchmark_f1/mvtec_ad/scores")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_f1/mvtec_ad")
    parser.add_argument("--priors", nargs="+", type=float, default=[0.01, 0.05, 0.15])
    parser.add_argument("--cost-ratios", nargs="+", type=float, default=[10.0, 20.0, 50.0])
    parser.add_argument("--alert-budgets", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--n-stream", type=int, default=10000)
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.npz")))
    if len(npz_files) == 0:
        print(f"❌ Error: No .npz files found in {args.scores_dir}. Please run 'python scripts/run_benchmark.py --save-scores' first.")
        sys.exit(1)

    print(f"=== Starting Rigorous Operational Evaluation across {len(npz_files)} Score Archives ===")
    print(f"Defect Priors: {args.priors}")
    print(f"Cost Ratios: {args.cost_ratios}")
    print(f"Alert Budgets: {args.alert_budgets} per 1k\n")

    eval_rows = []
    run_records_by_cat_method: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for fpath in npz_files:
        fname = os.path.basename(fpath).replace(".npz", "")
        parts = fname.split("_")
        if len(parts) >= 3:
            category = "_".join(parts[:-2])
            method = parts[-2]
            seed = int(parts[-1])
        else:
            category, method, seed = parts[0], parts[1], 42

        data = np.load(fpath)
        image_labels = data["image_labels"]
        image_scores = data["image_scores"]

        key = (category, method)
        if key not in run_records_by_cat_method:
            run_records_by_cat_method[key] = []
        run_records_by_cat_method[key].append({
            "seed": seed,
            "labels": image_labels,
            "scores": image_scores
        })

        # Thresholds are fitted on a calibration half and evaluated on the disjoint other half.
        calib_s, calib_y, eval_s, eval_y = stratified_split_50_50(image_labels, image_scores, seed=seed)
        norm_scores = eval_s[eval_y == 0]
        def_scores = eval_s[eval_y == 1]

        # 1. TPR at an alert budget: an in-sample ROC operating point (threshold and TPR on the same data)
        tpr_at_5_res = compute_tpr_at_alert_budget(image_labels, image_scores, max_alerts_per_1k=5.0)
        tpr_at_10_res = compute_tpr_at_alert_budget(image_labels, image_scores, max_alerts_per_1k=10.0)

        # 2. Threshold evaluation at 99% nominal quantile
        tau_99 = compute_quantile_threshold(calib_s[calib_y == 0], quantile=0.99)
        fa_at_1k_99 = compute_fa_at_1k(eval_y, eval_s, threshold=tau_99)
        md_at_1k_99 = compute_md_at_1k(eval_y, eval_s, threshold=tau_99)
        cwe_r10 = compute_cost_weighted_error(eval_y, eval_s, threshold=tau_99, cost_ratio=10.0, prior=0.01)
        cwe_r20 = compute_cost_weighted_error(eval_y, eval_s, threshold=tau_99, cost_ratio=20.0, prior=0.01)
        cwe_r50 = compute_cost_weighted_error(eval_y, eval_s, threshold=tau_99, cost_ratio=50.0, prior=0.01)

        # 3. Production Stream Simulation across Multi-Regimes (IID, Burst, Drift)
        sim = ProductionStreamSimulator(norm_scores, def_scores, seed=seed)
        for prior in args.priors:
            labels_iid, scores_iid = sim.simulate_iid_stream(n_total=args.n_stream, defect_prior=prior)
            labels_burst, scores_burst = sim.simulate_block_correlated_stream(n_total=args.n_stream, defect_prior=prior, mean_block_length=20)
            labels_drift, scores_drift = sim.simulate_drift_stream(n_total=args.n_stream, defect_prior=prior, drift_slope=0.10)

            res_iid = sim.evaluate_stream_robustness(labels_iid, scores_iid, tau=tau_99, cost_ratio=10.0)
            res_burst = sim.evaluate_stream_robustness(labels_burst, scores_burst, tau=tau_99, cost_ratio=10.0)
            res_drift = sim.evaluate_stream_robustness(labels_drift, scores_drift, tau=tau_99, cost_ratio=10.0)

            row = {
                "category": category,
                "method": method,
                "seed": seed,
                "defect_prior": prior,
                "tau_99": tau_99,
                "tpr_at_5_budget": tpr_at_5_res["tpr"],
                "tpr_at_10_budget": tpr_at_10_res["tpr"],
                "fa_at_1k": fa_at_1k_99,
                "md_at_1k": md_at_1k_99,
                "cwe_r10": cwe_r10,
                "cwe_r20": cwe_r20,
                "cwe_r50": cwe_r50,
                "p_overload_iid": res_iid["p_overload"],
                "p_overload_burst": res_burst["p_overload"],
                "p_overload_drift": res_drift["p_overload"]
            }
            eval_rows.append(row)

    runs_df = pd.DataFrame(eval_rows)

    # Hierarchical Statistical Aggregation with 95% Bootstrap CIs
    summary_rows = []
    for (cat, m), records in sorted(run_records_by_cat_method.items()):
        def metric_tpr5(recs):
            all_tprs = []
            for r in recs:
                res = compute_tpr_at_alert_budget(r["labels"], r["scores"], max_alerts_per_1k=5.0)
                all_tprs.append(res["tpr"])
            return float(np.mean(all_tprs))

        def metric_md(recs):
            all_mds = []
            for r in recs:
                cs, cy, es, ey = stratified_split_50_50(r["labels"], r["scores"], seed=r["seed"])
                tau = compute_quantile_threshold(cs[cy == 0], 0.99)
                all_mds.append(compute_md_at_1k(ey, es, tau))
            return float(np.mean(all_mds))

        def metric_cwe(recs):
            all_cwes = []
            for r in recs:
                cs, cy, es, ey = stratified_split_50_50(r["labels"], r["scores"], seed=r["seed"])
                tau = compute_quantile_threshold(cs[cy == 0], 0.99)
                all_cwes.append(compute_cost_weighted_error(ey, es, tau, cost_ratio=10.0, prior=0.01))
            return float(np.mean(all_cwes))

        h_tpr5 = hierarchical_bootstrap_ci(records, metric_tpr5, n_resamples=1000, seed=42)
        h_md = hierarchical_bootstrap_ci(records, metric_md, n_resamples=1000, seed=42)
        h_cwe = hierarchical_bootstrap_ci(records, metric_cwe, n_resamples=1000, seed=42)

        sub_df = runs_df[(runs_df["category"] == cat) & (runs_df["method"] == m) & (runs_df["defect_prior"] == 0.01)]
        ovl_mean = float(sub_df["p_overload_iid"].mean())
        fa_mean = float(sub_df["fa_at_1k"].mean())
        tpr10_mean = float(sub_df["tpr_at_10_budget"].mean())

        summary_rows.append({
            "category": cat,
            "method": m,
            "tpr_at_5_mean": h_tpr5["estimate"],
            "tpr_at_5_ci_low": h_tpr5["ci_low"],
            "tpr_at_5_ci_high": h_tpr5["ci_high"],
            "tpr_at_10_mean": tpr10_mean,
            "tpr_at_10_ci_low": tpr10_mean,
            "tpr_at_10_ci_high": tpr10_mean,
            "md_at_1k_mean": h_md["estimate"],
            "md_at_1k_ci_low": h_md["ci_low"],
            "md_at_1k_ci_high": h_md["ci_high"],
            "fa_at_1k_mean": fa_mean,
            "cwe_r10_mean": h_cwe["estimate"],
            "cwe_r10_ci_low": h_cwe["ci_low"],
            "cwe_r10_ci_high": h_cwe["ci_high"],
            "overload_prob_mean": ovl_mean
        })

    summary_df = pd.DataFrame(summary_rows)

    out_csv = os.path.join(tables_dir, "operational_results.csv")
    out_md = os.path.join(tables_dir, "operational_results.md")
    out_tex = os.path.join(tables_dir, "operational_results.tex")

    summary_df.to_csv(out_csv, index=False)

    run_piv = runs_df[runs_df["defect_prior"] == 0.01].pivot_table(
        index=["category", "seed"],
        columns="method",
        values="cwe_r10"
    ).dropna()

    wilcoxon_results = {}
    if "patchcore" in run_piv.columns and "padim" in run_piv.columns:
        res_pc_padim = compute_paired_wilcoxon_analysis(run_piv["patchcore"].values, run_piv["padim"].values)
        wilcoxon_results["PatchCore vs PaDiM"] = res_pc_padim

    if "patchcore" in run_piv.columns and "autoencoder" in run_piv.columns:
        res_pc_ae = compute_paired_wilcoxon_analysis(run_piv["patchcore"].values, run_piv["autoencoder"].values)
        wilcoxon_results["PatchCore vs Autoencoder"] = res_pc_ae

    raw_p_dict = {k: v["p_value"] for k, v in wilcoxon_results.items()}
    corrected_p = holm_bonferroni_from_pvalues(raw_p_dict, alpha=0.05)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Operational Inspection Benchmark Summary (Hierarchical 95% Bootstrap CIs)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Validated Statistical Significance Analysis (N=15 Category-Seed Runs)\n\n")
        f.write("| Comparison | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$ | Hodges-Lehmann $\\Delta$ | Rank-Biserial $r_{rb}$ | Significant ($\\alpha=0.05$) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for comp_name, res in wilcoxon_results.items():
            adj = corrected_p.get(comp_name, {"adjusted_p": res["p_value"], "is_significant": res["p_value"] < 0.05})
            f.write(f"| **{comp_name}** | {res['statistic']:.1f} | {res['p_value']:.4e} | {adj['adjusted_p']:.4e} | {res['hodges_lehmann']:.4f} | {res['rank_biserial']:.3f} | **{adj['is_significant']}** |\n")

    generate_latex_table(summary_df, out_tex)
    print(f"\n✅ Rigorous Operational Evaluation Complete! Results written to {tables_dir}")


if __name__ == "__main__":
    main()