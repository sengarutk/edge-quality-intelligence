# Coreset selection on synthetic Gaussian features (N=10,000, D=128, ratio 0.10)

Features are random normal vectors, not PatchCore descriptors; see coreset_scalability.md for real features.

| method                 |   runtime_sec |   speedup_vs_cpu |   peak_vram_mb |   coverage_radius |
|:-----------------------|--------------:|-----------------:|---------------:|------------------:|
| cpu_sequential_greedy  |     1.30973   |          1       |         0      |           14.9339 |
| gpu_unbatched_greedy   |     0.200377  |          6.53634 |        22.8892 |           14.9248 |
| gpu_batched_vectorized |     0.0748115 |         17.5071  |        23.1172 |           14.9247 |
| random_subsampling     |     0.0025599 |        511.634   |         0      |           15.3702 |