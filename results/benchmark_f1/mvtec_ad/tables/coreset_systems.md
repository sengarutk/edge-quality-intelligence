# PatchCore GPU Coreset Systems Benchmark (N=10,000, D=128, Ratio=0.10)

| method                 |   runtime_sec |   speedup_vs_cpu |   peak_vram_mb |   coverage_radius |
|:-----------------------|--------------:|-----------------:|---------------:|------------------:|
| cpu_sequential_greedy  |    1.2401     |          1       |         0      |           14.9339 |
| gpu_unbatched_greedy   |    0.129694   |          9.56168 |        22.8892 |           14.9248 |
| gpu_batched_vectorized |    0.0431775  |         28.7209  |        23.1172 |           14.9247 |
| random_subsampling     |    0.00078949 |       1570.76    |         0      |           15.3702 |