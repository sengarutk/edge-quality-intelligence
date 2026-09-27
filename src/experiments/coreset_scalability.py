"""Coreset subsampling scalability: every configuration is measured, none is extrapolated.

Features are real PatchCore patch descriptors (ResNet-18 layer2+layer3, 384-d) extracted
from MVTec AD training images and randomly projected to D dimensions. For each (N, D)
the four strategies of src/experiments/coreset_systems.py select M = 10% of N points and
we report wall time, peak GPU memory and the coverage radius max_x min_c ||x - c||
(lower is better; greedy k-center is a 2-approximation of the optimum).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.experiments.coreset_systems import (
    compute_coverage_radius,
    cpu_sequential_greedy,
    gpu_batched_vectorized,
    gpu_unbatched_greedy,
    random_subsampling,
)


def extract_patch_features(category: str = "metal_nut", data_root: str = "data/mvtec_ad", max_images: int = 120,
                           device: Optional[str] = None) -> np.ndarray:
    """Patch features (n_patches, 384) from up to ``max_images`` training images."""
    from torch.utils.data import DataLoader, Subset

    from src.data.mvtec import MVTecTrainNormal
    from src.models import PatchCore

    ds = MVTecTrainNormal(data_root, category, img_size=224)
    model = PatchCore(device=device)
    feats = [model._extract_multiscale_features(x).cpu() for x in
             DataLoader(Subset(ds, range(min(max_images, len(ds)))), batch_size=16)]
    return torch.cat(feats).numpy().astype(np.float32)


def run_coreset_scalability_sweep(
    sample_sizes: List[int] = [1000, 5000, 10000, 25000, 50000],
    feature_dims: List[int] = [64, 128, 256],
    coreset_ratio: float = 0.10,
    output_dir: str = "results/benchmark_f1/mvtec_ad",
    device_name: Optional[str] = None,
    features: Optional[np.ndarray] = None,
    seed: int = 0,
) -> pd.DataFrame:
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    pool = features if features is not None else extract_patch_features(device=str(device))
    rng = np.random.default_rng(seed)

    if device.type == "cuda":  # initialize the CUDA context outside the timed region
        gpu_batched_vectorized(torch.randn(500, 64, device=device), target_size=50, batch_k=25)
        torch.cuda.synchronize()

    records = []
    for dim in feature_dims:
        proj, _ = np.linalg.qr(rng.standard_normal((pool.shape[1], dim)))
        for n in sample_sizes:
            if n > pool.shape[0]:
                raise ValueError(f"Requested N={n} but only {pool.shape[0]} patches are available")
            x = (pool[rng.choice(pool.shape[0], n, replace=False)] @ proj).astype(np.float32)
            m = max(10, int(n * coreset_ratio))
            xt = torch.from_numpy(x).to(device)

            idx_cpu, t_cpu = cpu_sequential_greedy(x, target_size=m, seed=seed)
            idx_gu, t_gu, v_gu = gpu_unbatched_greedy(xt, target_size=m, seed=seed)
            idx_gb, t_gb, v_gb = gpu_batched_vectorized(xt, target_size=m, batch_k=50, seed=seed)
            idx_r, t_r = random_subsampling(n, target_size=m, seed=seed)

            records.append({
                "num_patches_N": n, "feature_dim_D": dim, "target_coreset_M": m,
                "time_cpu_greedy_sec": t_cpu, "time_gpu_greedy_sec": t_gu, "time_gpu_batched_sec": t_gb,
                "speedup_gpu_greedy_vs_cpu": t_cpu / t_gu, "speedup_gpu_batched_vs_cpu": t_cpu / t_gb,
                "peak_vram_gpu_greedy_mb": v_gu, "peak_vram_gpu_batched_mb": v_gb,
                "radius_cpu_greedy": compute_coverage_radius(x, idx_cpu),
                "radius_gpu_greedy": compute_coverage_radius(x, idx_gu),
                "radius_gpu_batched": compute_coverage_radius(x, idx_gb),
                "radius_random": compute_coverage_radius(x, idx_r),
            })
            print(f"N={n} D={dim}: cpu {t_cpu:.2f}s gpu {t_gu:.3f}s batched {t_gb:.4f}s")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(tables_dir, "coreset_scalability.csv"), index=False)
    with open(os.path.join(tables_dir, "coreset_scalability.md"), "w", encoding="utf-8") as f:
        f.write("# Coreset subsampling scalability (all values measured)\n\n")
        f.write(df.to_markdown(index=False))

    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Coreset subsampling on real PatchCore patch features ($M = 0.1N$). All times are measured. "
        "Radius is the coverage radius $\\max_x \\min_c \\lVert x - c \\rVert$ (lower is better).}",
        "\\label{tab:coreset_scalability}", "\\resizebox{0.95\\textwidth}{!}{%", "\\begin{tabular}{rrrrrrrr}", "\\toprule",
        "$N$ & $D$ & CPU greedy (s) & GPU greedy (s) & GPU batched (s) & Radius greedy & Radius batched & Radius random \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{int(r.num_patches_N):,} & {int(r.feature_dim_D)} & {r.time_cpu_greedy_sec:.3f} & "
                     f"{r.time_gpu_greedy_sec:.3f} & {r.time_gpu_batched_sec:.4f} & {r.radius_cpu_greedy:.3f} & "
                     f"{r.radius_gpu_batched:.3f} & {r.radius_random:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    with open(os.path.join(tables_dir, "coreset_scalability.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return df
