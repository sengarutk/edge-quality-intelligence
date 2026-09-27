import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch


def cpu_sequential_greedy(features: np.ndarray, target_size: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Standard sequential greedy k-center on CPU.
    """
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    n_samples = features.shape[0]
    target_size = min(target_size, n_samples)
    
    first_idx = int(rng.randint(0, n_samples))
    selected_indices = [first_idx]
    
    # Initialize min distances to first chosen point
    diff = features - features[first_idx]
    min_distances = np.sum(diff ** 2, axis=1)
    
    for _ in range(1, target_size):
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)
        diff = features - features[next_idx]
        new_dist = np.sum(diff ** 2, axis=1)
        min_distances = np.minimum(min_distances, new_dist)
        
    elapsed = time.perf_counter() - t0
    return np.array(selected_indices, dtype=int), elapsed


def gpu_unbatched_greedy(features: torch.Tensor, target_size: int, seed: int = 42) -> Tuple[np.ndarray, float, float]:
    """
    Unbatched greedy k-center on GPU (point-by-point update).
    """
    device = features.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    n_samples = features.shape[0]
    target_size = min(target_size, n_samples)
    
    first_idx = int(torch.randint(0, n_samples, (1,), device=device).item())
    selected_indices = [first_idx]
    
    # Distance to first point
    diff = features - features[first_idx]
    min_distances = torch.sum(diff ** 2, dim=1)
    
    for _ in range(1, target_size):
        next_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(next_idx)
        diff = features - features[next_idx]
        new_dist = torch.sum(diff ** 2, dim=1)
        min_distances = torch.minimum(min_distances, new_dist)
        
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak_vram_mb = 0.0
        
    elapsed = time.perf_counter() - t0
    return np.array(selected_indices, dtype=int), elapsed, peak_vram_mb


def gpu_batched_vectorized(
    features: torch.Tensor,
    target_size: int,
    batch_k: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, float, float]:
    """
    High-performance batched vectorized k-center on GPU utilizing algebraic distance expansion:
      ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    and top-k batch selection without serial scalar synchronization.
    """
    device = features.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    n_samples = features.shape[0]
    target_size = min(target_size, n_samples)
    
    first_idx = int(torch.randint(0, n_samples, (1,), device=device).item())
    selected_indices = [first_idx]
    
    x_sq = torch.sum(features ** 2, dim=1)
    
    # Initial distance
    start_pt = features[first_idx]
    start_sq_norm = torch.sum(start_pt ** 2)
    dot_start = torch.mv(features, start_pt)
    min_distances_sq = torch.clamp(x_sq + start_sq_norm - 2.0 * dot_start, min=0.0)
    
    while len(selected_indices) < target_size:
        curr_batch_k = min(batch_k, target_size - len(selected_indices))
        if curr_batch_k == 1:
            next_idx = int(torch.argmax(min_distances_sq).item())
            selected_indices.append(next_idx)
            query_pts = features[next_idx:next_idx+1]
        else:
            _, top_idx = torch.topk(min_distances_sq, k=curr_batch_k)
            selected_indices.extend(top_idx.cpu().tolist())
            query_pts = features[top_idx]
            
        dots = torch.matmul(features, query_pts.T)
        q_sq = torch.sum(query_pts ** 2, dim=1).unsqueeze(0)
        dist_sqs = torch.clamp(x_sq.unsqueeze(1) + q_sq - 2.0 * dots, min=0.0)
        min_distances_sq = torch.minimum(min_distances_sq, torch.min(dist_sqs, dim=1).values)
        
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak_vram_mb = 0.0
        
    elapsed = time.perf_counter() - t0
    return np.array(selected_indices[:target_size], dtype=int), elapsed, peak_vram_mb


def random_subsampling(n_samples: int, target_size: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Uniform random subsampling baseline.
    """
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    target_size = min(target_size, n_samples)
    indices = rng.choice(n_samples, size=target_size, replace=False)
    elapsed = time.perf_counter() - t0
    return indices, elapsed


def compute_coverage_radius(features: np.ndarray, selected_indices: np.ndarray, chunk_size: int = 1024) -> float:
    """
    Computes minimax coverage radius: max_{x in X} min_{c in C} ||x - c|| using chunked torch.cdist.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(features).to(device)
    C = X[selected_indices]
    max_min_dist = 0.0
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk_size):
            chunk = X[i:i + chunk_size]
            dists = torch.cdist(chunk, C, p=2.0)
            min_d, _ = torch.min(dists, dim=1)
            max_min_dist = max(max_min_dist, float(torch.max(min_d).item()))
    return float(max_min_dist)


def run_coreset_systems_benchmark(
    n_samples: int = 5000,
    dim: int = 128,
    sampling_ratio: float = 0.10,
    device_name: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Executes a multi-strategy coreset reduction benchmark on synthetic feature tensors.
    """
    target_size = max(10, int(n_samples * sampling_ratio))
    rng = np.random.RandomState(42)
    feats_np = rng.randn(n_samples, dim).astype(np.float32)
    
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    feats_torch = torch.from_numpy(feats_np).to(device)
    
    # 1. CPU Sequential Greedy
    idx_cpu, t_cpu = cpu_sequential_greedy(feats_np, target_size=target_size)
    radius_cpu = compute_coverage_radius(feats_np, idx_cpu)
    
    # 2. GPU Unbatched Greedy
    idx_gpu_unbatch, t_gpu_unbatch, vram_unbatch = gpu_unbatched_greedy(feats_torch, target_size=target_size)
    radius_gpu_unbatch = compute_coverage_radius(feats_np, idx_gpu_unbatch)
    
    # 3. GPU Batched Vectorized (Ours)
    idx_gpu_batch, t_gpu_batch, vram_batch = gpu_batched_vectorized(feats_torch, target_size=target_size, batch_k=50)
    radius_gpu_batch = compute_coverage_radius(feats_np, idx_gpu_batch)
    
    # 4. Random Baseline
    idx_rand, t_rand = random_subsampling(n_samples, target_size=target_size)
    radius_rand = compute_coverage_radius(feats_np, idx_rand)
    
    speedup_gpu_batch = t_cpu / max(1e-6, t_gpu_batch)
    speedup_gpu_unbatch = t_cpu / max(1e-6, t_gpu_unbatch)
    
    return {
        "cpu_sequential_greedy": {
            "time_sec": float(t_cpu),
            "speedup": 1.0,
            "peak_vram_mb": 0.0,
            "coverage_radius": radius_cpu
        },
        "gpu_unbatched_greedy": {
            "time_sec": float(t_gpu_unbatch),
            "speedup": float(speedup_gpu_unbatch),
            "peak_vram_mb": float(vram_unbatch),
            "coverage_radius": radius_gpu_unbatch
        },
        "gpu_batched_vectorized": {
            "time_sec": float(t_gpu_batch),
            "speedup": float(speedup_gpu_batch),
            "peak_vram_mb": float(vram_batch),
            "coverage_radius": radius_gpu_batch
        },
        "random_subsampling": {
            "time_sec": float(t_rand),
            "speedup": float(t_cpu / max(1e-6, t_rand)),
            "peak_vram_mb": 0.0,
            "coverage_radius": radius_rand
        }
    }