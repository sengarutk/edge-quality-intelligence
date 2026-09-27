import numpy as np
import torch
import pytest

from src.experiments.coreset_systems import (
    cpu_sequential_greedy,
    gpu_unbatched_greedy,
    gpu_batched_vectorized,
    random_subsampling,
    compute_coverage_radius,
    run_coreset_systems_benchmark
)


def test_coreset_reduction_methods():
    """
    Verifies that all coreset reduction algorithms select valid indices and maintain coverage.
    """
    n_samples = 500
    dim = 32
    target_size = 50
    
    rng = np.random.RandomState(42)
    feats_np = rng.randn(n_samples, dim).astype(np.float32)
    feats_torch = torch.from_numpy(feats_np)
    
    # 1. CPU Sequential
    idx_cpu, t_cpu = cpu_sequential_greedy(feats_np, target_size=target_size)
    assert len(idx_cpu) == target_size
    assert len(np.unique(idx_cpu)) == target_size
    rad_cpu = compute_coverage_radius(feats_np, idx_cpu)
    
    # 2. GPU Unbatched
    idx_gpu_u, t_gpu_u, _ = gpu_unbatched_greedy(feats_torch, target_size=target_size)
    assert len(idx_gpu_u) == target_size
    assert len(np.unique(idx_gpu_u)) == target_size
    rad_gpu_u = compute_coverage_radius(feats_np, idx_gpu_u)
    
    # 3. GPU Batched Vectorized
    idx_gpu_b, t_gpu_b, _ = gpu_batched_vectorized(feats_torch, target_size=target_size, batch_k=10)
    assert len(idx_gpu_b) == target_size
    assert len(np.unique(idx_gpu_b)) == target_size
    rad_gpu_b = compute_coverage_radius(feats_np, idx_gpu_b)
    
    # 4. Random Subsampling
    idx_rand, t_rand = random_subsampling(n_samples, target_size=target_size)
    assert len(idx_rand) == target_size
    rad_rand = compute_coverage_radius(feats_np, idx_rand)
    
    # Greedy methods must achieve tighter (or comparable) minimax coverage radius than random
    assert rad_cpu <= rad_rand * 1.2
    assert rad_gpu_b <= rad_rand * 1.2


def test_run_coreset_systems_benchmark():
    """
    Verifies end-to-end benchmark execution and speedup dictionary structure.
    """
    results = run_coreset_systems_benchmark(n_samples=200, dim=16, sampling_ratio=0.10, device_name="cpu")
    assert "cpu_sequential_greedy" in results
    assert "gpu_batched_vectorized" in results
    assert "random_subsampling" in results
    for m, vals in results.items():
        assert "time_sec" in vals
        assert "speedup" in vals
        assert "coverage_radius" in vals