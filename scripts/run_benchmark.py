import argparse
import os
import sys
import time
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import seed_everything
from src.mvtec import MVTecTrainNormal
from src.models import ConvAutoencoder, PaDiM, PatchCore
from src.robustness.dataset import CorruptedMVTecTest
from src.robustness.evaluator import RobustnessEvaluator
from src.benchmarking.profiler import CUDAPerformanceProfiler


def get_model(method_name: str, device: str, seed: int = 42):
    """Build a detector whose stochastic components depend on ``seed``."""
    method = method_name.lower()
    if method == "patchcore":
        return PatchCore(device=device, seed=seed)
    elif method == "padim":
        return PaDiM(device=device, random_seed=seed)
    elif method == "autoencoder":
        return ConvAutoencoder(device=device, epochs=20)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def main():
    parser = argparse.ArgumentParser(description="Master Multi-Seed Benchmark Runner")
    parser.add_argument("--categories", nargs="+",
                        default=["bottle", "cable", "carpet", "grid", "hazelnut", "leather", "metal_nut"])
    parser.add_argument("--methods", nargs="+", default=["patchcore", "padim", "autoencoder"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-robustness", action="store_true", default=False,
                        help="Run the 18-condition corruption stress test (about 18x slower)")
    parser.add_argument("--no-run-robustness", dest="run_robustness", action="store_false")
    parser.add_argument("--run-profiling", action="store_true", default=True)
    parser.add_argument("--no-run-profiling", dest="run_profiling", action="store_false")
    parser.add_argument("--save-scores", action="store_true", default=True, help="Persist .npz score archives")
    parser.add_argument("--no-save-scores", dest="save_scores", action="store_false")
    parser.add_argument("--scores-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Determine default output directory based on dataset root
    if args.output_dir is None:
        if "synthetic" in args.data_root.lower() or "mock" in args.data_root.lower():
            args.output_dir = "results/synthetic_validation"
        else:
            args.output_dir = "results/benchmark_f1/mvtec_ad"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tables_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")
    scores_dir = args.scores_dir if args.scores_dir is not None else os.path.join(args.output_dir, "scores")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    if args.save_scores:
        os.makedirs(scores_dir, exist_ok=True)

    master_csv_path = os.path.join(tables_dir, "runs_master.csv")
    if os.path.exists(master_csv_path):
        os.remove(master_csv_path)  # never mix rows from earlier invocations
    profiler = CUDAPerformanceProfiler(warmup_runs=50, active_runs=300, device=device)

    rows = []
    print(f"=== Starting Benchmark on Device: {device} ===")
    print(f"Output Directory: {args.output_dir}")
    print(f"Scores Directory: {scores_dir}")
    print(f"Categories: {args.categories}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}\n")

    for seed in args.seeds:
        for cat in args.categories:
            for method_name in args.methods:
                print(f"--> Running [Seed: {seed}] [Cat: {cat}] [Method: {method_name}]")
                seed_everything(seed)

                # 1. Fit Model
                train_ds = MVTecTrainNormal(args.data_root, cat)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

                t0 = time.time()
                model = get_model(method_name, device, seed)
                model.fit(train_loader)
                fit_time_s = time.time() - t0

                # 2. Dual-Latency Hardware Profiling (T_model & T_e2e)
                prof_results = {}
                if args.run_profiling:
                    prof_results = profiler.profile_dual(model, input_shape=(1, 3, 256, 256))

                # 3. Clean Evaluation & Score Extraction
                evaluator = RobustnessEvaluator(model, args.data_root, cat, device=device)
                test_ds = CorruptedMVTecTest(args.data_root, cat, corruption_type=None)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
                
                clean_image_labels, clean_image_scores, clean_gt_masks, clean_pixel_amaps = evaluator.predict_split(test_loader)
                clean_metrics = evaluator.evaluate_predictions(clean_image_labels, clean_image_scores, clean_gt_masks, clean_pixel_amaps)

                # Persist score archive if enabled
                if args.save_scores:
                    score_file = os.path.join(scores_dir, f"{cat}_{method_name}_{seed}.npz")
                    np.savez_compressed(
                        score_file,
                        image_scores=clean_image_scores,
                        image_labels=clean_image_labels,
                        pixel_amaps=clean_pixel_amaps.astype(np.float32),
                        ground_truth_masks=clean_gt_masks.astype(np.float32)
                    )

                # 4. Robustness Stress-Test
                mrd_auroc, mrd_aupro = float("nan"), float("nan")
                signed_drop_auroc, signed_drop_aupro = float("nan"), float("nan")
                if args.run_robustness:
                    stress_res = evaluator.run_full_stress_test(batch_size=args.batch_size)
                    mrd_auroc = stress_res["mrd_image_auroc"]
                    mrd_aupro = stress_res["mrd_aupro"]
                    signed_drop_auroc = stress_res.get("mean_performance_change_auroc", mrd_auroc)
                    signed_drop_aupro = stress_res.get("mean_performance_change_aupro", mrd_aupro)

                row = {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "seed": seed,
                    "category": cat,
                    "method": method_name,
                    "device": device,
                    "fit_time_s": fit_time_s,
                    "image_auroc": clean_metrics["image_auroc"],
                    "image_ap": clean_metrics["image_ap"],
                    "pixel_auroc": clean_metrics["pixel_auroc"],
                    "pixel_ap": clean_metrics["pixel_ap"],
                    "aupro": clean_metrics["aupro"],
                    "max_f1": clean_metrics["max_f1"],
                    "optimal_threshold": clean_metrics.get("optimal_threshold", 0.0),
                    "oracle_max_f1": clean_metrics["oracle_max_f1"],
                    "oracle_threshold": clean_metrics["oracle_threshold"],
                    "precision_at_optimal": clean_metrics.get("precision_at_optimal", 0.0),
                    "recall_at_optimal": clean_metrics.get("recall_at_optimal", 0.0),
                    "ece": clean_metrics["ece"],
                    "mrd_image_auroc": mrd_auroc,
                    "mrd_aupro": mrd_aupro,
                    "mean_performance_change_auroc": signed_drop_auroc,
                    "mean_performance_change_aupro": signed_drop_aupro,
                    "p50_model_ms": prof_results.get("p50_model_ms", float("nan")),
                    "p95_model_ms": prof_results.get("p95_model_ms", float("nan")),
                    "fps_model": prof_results.get("fps_model", float("nan")),
                    "p50_e2e_ms": prof_results.get("p50_e2e_ms", float("nan")),
                    "p95_e2e_ms": prof_results.get("p95_e2e_ms", float("nan")),
                    "fps_e2e": prof_results.get("fps_e2e", float("nan")),
                    "p50_latency_ms": prof_results.get("p50_model_ms", float("nan")),
                    "fps": prof_results.get("fps_model", float("nan")),
                    "peak_vram_mb": prof_results.get("peak_vram_mb", float("nan")),
                }
                rows.append(row)
                df_row = pd.DataFrame([row])
                write_header = not os.path.exists(master_csv_path)
                df_row.to_csv(master_csv_path, mode="a", header=write_header, index=False)
                print(f"    Done: Image AUROC = {row['image_auroc']:.4f} | AU-PRO = {row['aupro']:.4f} | MRD = {row['mrd_image_auroc']:.4f}")

    # Multi-seed aggregate summary table
    df_all = pd.DataFrame(rows)
    group_cols = ["category", "method"]
    summary_df = df_all.groupby(group_cols).agg(
        image_auroc_mean=("image_auroc", "mean"),
        image_auroc_std=("image_auroc", "std"),
        aupro_mean=("aupro", "mean"),
        aupro_std=("aupro", "std"),
        pixel_auroc_mean=("pixel_auroc", "mean"),
        pixel_auroc_std=("pixel_auroc", "std"),
        mrd_mean=("mrd_image_auroc", "mean"),
        mrd_std=("mrd_image_auroc", "std"),
        p50_model_ms=("p50_model_ms", "mean"),
        p50_e2e_ms=("p50_e2e_ms", "mean"),
        fps_model=("fps_model", "mean"),
        fps_e2e=("fps_e2e", "mean"),
        peak_vram_mb=("peak_vram_mb", "mean"),
    ).reset_index()

    summary_csv = os.path.join(tables_dir, "summary_multiseed.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_md = os.path.join(tables_dir, "summary_multiseed.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Master Multi-Seed Benchmark Summary (mean ± std)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

    print(f"\n=== Benchmark Complete! Saved summaries to {tables_dir} ===")


if __name__ == "__main__":
    main()
