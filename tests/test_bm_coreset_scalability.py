import numpy as np

from src.experiments.coreset_scalability import run_coreset_scalability_sweep


def test_coreset_scalability_sweep_is_fully_measured(tmp_path):
    feats = np.random.default_rng(0).standard_normal((3000, 384)).astype(np.float32)
    df = run_coreset_scalability_sweep(sample_sizes=[500, 1000], feature_dims=[64, 128], coreset_ratio=0.10,
                                       output_dir=str(tmp_path), features=feats)
    assert len(df) == 4
    for col in ("time_cpu_greedy_sec", "time_gpu_greedy_sec", "time_gpu_batched_sec", "radius_cpu_greedy",
                "radius_gpu_batched", "radius_random"):
        assert np.all(df[col] > 0)
    # greedy k-center (2-approximation) covers the data at least as well as random subsampling
    assert np.all(df["radius_cpu_greedy"] <= df["radius_random"] + 1e-6)
    assert np.all(df["radius_gpu_greedy"] <= df["radius_random"] + 1e-6)


def test_requesting_more_points_than_available_fails(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        run_coreset_scalability_sweep(sample_sizes=[5000], feature_dims=[64], output_dir=str(tmp_path),
                                      features=np.zeros((100, 384), np.float32))
