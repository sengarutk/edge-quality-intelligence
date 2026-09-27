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
    if True:
        method_labels: Dict[str, List[int]] = {"patchcore": [], "padim": [], "autoencoder": []}
        method_scores: Dict[str, List[float]] = {"patchcore": [], "padim": [], "autoencoder": []}
        overload_rows = []

        all_y_list = []
        all_s_list = []

        for fpath in npz_files:
            fname = os.path.basename(fpath).replace(".npz", "")
            parts = fname.split("_")
            if len(parts) >= 3:
                m_name = parts[-2]
            else:
                m_name = parts[1]

            m_lower = str(m_name).lower()
            if m_lower not in method_labels:
                method_labels[m_lower] = []
                method_scores[m_lower] = []

            data = np.load(fpath)
            y = data["image_labels"]
            s = data["image_scores"]

            s_norm = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8)

            method_labels[m_lower].extend(y.tolist())
            method_scores[m_lower].extend(s_norm.tolist())

            if m_lower == "patchcore":
                all_y_list.extend(y.tolist())
                all_s_list.extend(s_norm.tolist())

            norm_s = s_norm[y == 0]
            def_s = s_norm[y == 1]
            if len(norm_s) > 0 and len(def_s) > 0:
                sim = ProductionStreamSimulator(norm_s, def_s, seed=42)
                tau_99 = compute_quantile_threshold(norm_s, 0.99)
                for prior in [0.01, 0.05, 0.15]:
                    _, stream_s = sim.simulate_stream(n_total=5000, defect_prior=prior)
                    alerts = (stream_s >= tau_99).astype(int)
                    ovl = compute_operator_overload(alerts, operator_capacity_per_window=60, window_size=1000)
                    overload_rows.append({
                        "method": m_lower,
                        "defect_prior": prior,
                        "overload_probability": ovl["overload_probability"]
                    })

        method_data = {}
        for m in method_labels:
            if len(method_labels[m]) > 0:
                method_data[m] = (np.array(method_labels[m]), np.array(method_scores[m]))

        all_labels = np.array(all_y_list)
        all_scores = np.array(all_s_list)

    # 1. FA vs MD Tradeoff
    p1 = os.path.join(args.output_dir, "fa_vs_md_tradeoff.png")
    plot_fa_vs_md_tradeoff(method_data, p1)
    print(f"✅ Generated: {p1}")

    # 2. TPR vs Alert Budget
    p2 = os.path.join(args.output_dir, "tpr_vs_alert_budget.png")
    plot_tpr_vs_alert_budget(method_data, p2)
    print(f"✅ Generated: {p2}")

    # 3. Cost-Weighted Error Curves
    p3 = os.path.join(args.output_dir, "cost_weighted_error_curves.png")
    plot_cost_weighted_error_curves(method_data, p3)
    print(f"✅ Generated: {p3}")

    # 4. Operator Review Overload
    p4 = os.path.join(args.output_dir, "operator_review_overload.png")
    plot_operator_review_overload(pd.DataFrame(overload_rows), p4)
    print(f"✅ Generated: {p4}")

    # 5. CCT Empirical Risk Tradeoff
    p5 = os.path.join(args.output_dir, "cct_cost_tradeoff.png")
    plot_cct_cost_tradeoff(all_scores, all_labels, p5, cost_ratios=[10.0, 20.0, 50.0], prior=0.01)
    print(f"✅ Generated: {p5}")

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