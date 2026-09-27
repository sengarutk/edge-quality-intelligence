import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pandas as pd

from src.experiments.cct_ablation import run_cct_out_of_sample_ablation
from src.experiments.decision_changes import run_decision_change_analysis
from src.experiments.coreset_scalability import run_coreset_scalability_sweep


def main():
    parser = argparse.ArgumentParser(description="Cost-Calibrated Thresholding (CCT), Decision Changes, and Coreset Scalability Runner")
    parser.add_argument("--scores-dir", type=str, default="results/benchmark_f1/mvtec_ad/scores")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_f1/mvtec_ad")
    parser.add_argument("--coreset-ratio", type=float, default=0.10)
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print("=== 1. Running 50/50 Out-of-Sample Cost-Calibrated Thresholding (CCT) Ablation ===")
    cct_df = run_cct_out_of_sample_ablation(
        scores_dir=args.scores_dir,
        output_dir=args.output_dir,
        cost_ratios=[10.0, 20.0, 50.0],
        max_alerts_per_1k=5.0
    )
    print(f"✅ CCT Ablation generated across {len(cct_df)} runs.\n")

    print("=== 2. Running Operational Decision-Change Attribution Analysis ===")
    dec_df = run_decision_change_analysis(
        scores_dir=args.scores_dir,
        output_dir=args.output_dir
    )
    print(f"✅ Decision-Change Attribution generated across {len(dec_df)} runs.\n")

    print("=== 3. Running Systems Coreset Scalability Benchmark (N in [1k..50k], D in [64, 128, 256]) ===")
    scale_df = run_coreset_scalability_sweep(
        sample_sizes=[1000, 5000, 10000, 25000, 50000],
        feature_dims=[64, 128, 256],
        coreset_ratio=args.coreset_ratio,
        output_dir=args.output_dir
    )
    print(f"✅ Coreset Scalability Benchmark generated across {len(scale_df)} configurations.\n")

    print("=== ✅ All CCT and Scalability Experiments Successfully Completed ===")


if __name__ == "__main__":
    main()