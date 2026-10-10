import argparse
import os
import sys
import glob
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.publication_plots import (
    plot_fa_vs_md_tradeoff,
    plot_tpr_vs_alert_budget,
    plot_cost_weighted_error_curves,
    plot_operator_review_overload,
    plot_cct_cost_tradeoff,
    plot_coreset_scalability,
    plot_decision_confusion_shifts
)
from src.metrics.image_metrics import compute_quantile_threshold
from src.metrics.operational import compute_operator_overload
from src.experiments.operational_eval import ProductionStreamSimulator


def main():
    parser = argparse.ArgumentParser(description="Dedicated Operational and CCT Publication Plot Renderer")
    parser.add_argument("--scores-dir", type=str, default="results/benchmark_f1/mvtec_ad/scores")
    parser.add_argument("--tables-dir", type=str, default="results/benchmark_f1/mvtec_ad/tables")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_f1/mvtec_ad/figures/operational")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.npz")))
    if len(npz_files) == 0:
        raise SystemExit(f"No score archives in {args.scores_dir}; run scripts/run_benchmark.py --save-scores first.")

    # Every curve is computed per run (one category, method and seed) with that run's own raw scores,
    # and then averaged over runs. Scores are never normalized or pooled across categories.
    budgets = np.linspace(1.0, 50.0, 50)          # allowed false alarms per 1,000 good parts
    runs = {}
    for fpath in npz_files:
        parts = os.path.basename(fpath)[:-4].split("_")
        method = parts[-2].lower()
        data = np.load(fpath)
        runs.setdefault(method, []).append((parts, data["image_labels"].astype(int), data["image_scores"].astype(float)))

    fa_md_rows, tpr_rows, cwe_rows, overload_rows = [], [], [], []
    for method, items in runs.items():
        tprs, fas = [], []
        for _, y, s_ in items:
            nom, dfc = s_[y == 0], s_[y == 1]
            taus = [np.quantile(nom, 1.0 - b / 1000.0) for b in budgets]
            tprs.append([float(np.mean(dfc >= t)) for t in taus])
            fas.append([1000.0 * float(np.mean(nom >= t)) for t in taus])
            if method == "patchcore":
                for r in (10.0, 20.0, 50.0):
                    cwe_rows += [{"cost_ratio": r, "alert_budget": b, "cwe": 0.99 * np.mean(nom >= t) + 0.01 * r * np.mean(dfc < t)}
                                 for b, t in zip(budgets, taus)]
            sim = ProductionStreamSimulator(nom, dfc, seed=42)
            tau_99 = compute_quantile_threshold(nom, 0.99)
            for prior in [0.01, 0.05, 0.15]:
                _, stream_s = sim.simulate_stream(n_total=5000, defect_prior=prior)
                ovl = compute_operator_overload((stream_s >= tau_99).astype(int), operator_capacity_per_window=60,
                                                window_size=1000)
                overload_rows.append({"method": method, "defect_prior": prior,
                                      "overload_probability": ovl["overload_probability"]})
        tpr_m, fa_m = np.mean(tprs, axis=0), np.mean(fas, axis=0)
        tpr_rows += [{"method": method, "alert_budget": b, "tpr": t} for b, t in zip(budgets, tpr_m)]
        fa_md_rows += [{"method": method, "fa_at_1k": f, "md_at_1k": 1000.0 * (1.0 - t)} for f, t in zip(fa_m, tpr_m)]

    # 1. FA vs MD (in-sample on the test set: thresholds and rates from the same data)
    p1 = os.path.join(args.output_dir, "fa_vs_md_tradeoff.png")
    plot_fa_vs_md_tradeoff(pd.DataFrame(fa_md_rows), p1)
    print(f"Generated: {p1}")

    # 2. TPR vs alert budget (mean over runs)
    p2 = os.path.join(args.output_dir, "tpr_vs_alert_budget.png")
    plot_tpr_vs_alert_budget(pd.DataFrame(tpr_rows), p2, max_budget=50.0)
    print(f"Generated: {p2}")

    # 3. Expected cost (prior 0.01) vs alert budget for PatchCore, mean over runs
    p3 = os.path.join(args.output_dir, "cost_weighted_error_curves.png")
    cwe_df = pd.DataFrame(cwe_rows).groupby(["cost_ratio", "alert_budget"], as_index=False)["cwe"].mean()
    plot_cost_weighted_error_curves(cwe_df.rename(columns={"alert_budget": "threshold"}), p3)
    print(f"Generated: {p3}")

    # 4. Operator review overload
    p4 = os.path.join(args.output_dir, "operator_review_overload.png")
    plot_operator_review_overload(pd.DataFrame(overload_rows), p4)
    print(f"Generated: {p4}")

    # 5. CCT risk curve for one run (raw scores of a single category; not pooled)
    example = os.path.join(args.scores_dir, "metal_nut_patchcore_42.npz")
    if os.path.exists(example):
        d = np.load(example)
        p5 = os.path.join(args.output_dir, "cct_cost_tradeoff.png")
        plot_cct_cost_tradeoff(d["image_scores"], d["image_labels"], p5, cost_ratios=[10.0, 20.0, 50.0], prior=0.01)
        print(f"Generated: {p5} (metal_nut, PatchCore, seed 42)")

    # 6. Coreset Scalability Log-Log Scaling
    scale_csv = os.path.join(args.tables_dir, "coreset_scalability.csv")
    if os.path.exists(scale_csv):
        scale_df = pd.read_csv(scale_csv)
        p6 = os.path.join(args.output_dir, "coreset_scalability.png")
        plot_coreset_scalability(scale_df, p6)
        print(f"✅ Generated: {p6}")

    # 7. Decision Confusion Shifts
    dec_csv = os.path.join(args.tables_dir, "decision_changes.csv")
    if os.path.exists(dec_csv):
        dec_df = pd.read_csv(dec_csv)
        p7 = os.path.join(args.output_dir, "decision_confusion_shifts.png")
        plot_decision_confusion_shifts(dec_df, p7)
        print(f"✅ Generated: {p7}")

    print("\n✅ All Operational and CCT Publication Figures generated successfully!")


if __name__ == "__main__":
    main()