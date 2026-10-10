import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import argparse
import numpy as np
import pandas as pd
import torch

from src.experiments.aggregation_ablation import run_aggregation_ablation
from src.experiments.coreset_systems import run_coreset_systems_benchmark


def main():
    parser = argparse.ArgumentParser(description="Image Aggregation and Coreset Systems Ablations Orchestrator")
    parser.add_argument("--scores-dir", type=str, default="results/benchmark_f1/mvtec_ad/scores")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_f1/mvtec_ad")
    parser.add_argument("--n-samples", type=int, default=10000)
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print("=== 1. Running PatchCore GPU Coreset Vectorization Systems Benchmark ===")
    coreset_results = run_coreset_systems_benchmark(
        n_samples=args.n_samples,
        dim=128,
        sampling_ratio=0.10
    )

    coreset_rows = []
    for method, metrics in coreset_results.items():
        coreset_rows.append({
            "method": method,
            "runtime_sec": metrics["time_sec"],
            "speedup_vs_cpu": metrics["speedup"],
            "peak_vram_mb": metrics["peak_vram_mb"],
            "coverage_radius": metrics["coverage_radius"]
        })

    coreset_df = pd.DataFrame(coreset_rows)
    coreset_csv = os.path.join(tables_dir, "coreset_systems.csv")
    coreset_md = os.path.join(tables_dir, "coreset_systems.md")
    coreset_df.to_csv(coreset_csv, index=False)

    with open(coreset_md, "w", encoding="utf-8") as f:
        f.write(f"# Coreset selection on synthetic Gaussian features (N={args.n_samples:,}, D=128, ratio 0.10)\n\n"
                "Features are random normal vectors, not PatchCore descriptors; see coreset_scalability.md for real features.\n\n")
        f.write(coreset_df.to_markdown(index=False))

    print(f"✅ Coreset Systems Ablation saved to {coreset_csv}")
    print(coreset_df.to_string(index=False))

    print("\n=== 2. Running Image-Level Metric Aggregation Ablation ===")
    npz_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.npz")))
    
    agg_rows = []
    if len(npz_files) > 0:
        for fpath in npz_files:
            fname = os.path.basename(fpath).replace(".npz", "")
            parts = fname.split("_")
            if len(parts) >= 3:
                cat, meth, seed = "_".join(parts[:-2]), parts[-2], int(parts[-1])
            else:
                cat, meth, seed = parts[0], parts[1], 42
                
            data = np.load(fpath)
            labels = data["image_labels"]
            # Real (Gaussian-smoothed) pixel anomaly maps written by scripts/run_benchmark.py.
            if "pixel_amaps" not in data:
                raise KeyError(f"{fpath} has no pixel_amaps; rerun scripts/run_benchmark.py --save-scores")
            amaps = data["pixel_amaps"]
            res = run_aggregation_ablation(amaps, labels)
            for strat, m in res.items():
                agg_rows.append({
                    "category": cat,
                    "method": meth,
                    "seed": seed,
                    "aggregation_rule": strat,
                    "image_auroc": m["image_auroc"],
                    "image_ap": m["image_ap"]
                })
    else:
        raise SystemExit(f"No score archives in {args.scores_dir}; run scripts/run_benchmark.py --save-scores first.")

    agg_df = pd.DataFrame(agg_rows)
    agg_summary = agg_df.groupby("aggregation_rule").agg({
        "image_auroc": ["mean", "std"],
        "image_ap": ["mean", "std"]
    }).reset_index()

    agg_csv = os.path.join(tables_dir, "aggregation_ablation.csv")
    agg_md = os.path.join(tables_dir, "aggregation_ablation.md")
    agg_df.to_csv(agg_csv, index=False)

    with open(agg_md, "w", encoding="utf-8") as f:
        f.write("# Image-score aggregation of the stored pixel anomaly maps (pixel_amaps, already smoothed with sigma 4)\n\n")
        f.write(agg_summary.to_markdown(index=False))

    print(f"\n✅ Image Aggregation Ablation saved to {agg_csv}")
    print(agg_summary.to_string())


if __name__ == "__main__":
    main()