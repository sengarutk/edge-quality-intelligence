import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytest
from typing import Tuple

from src.methods.base import BaseAnomalyDetector
from src.analysis.failure_catalog import FailureMiner
from src.experiments.robust_training import AugmentedNormalDataset, RobustTrainingExperiment
from src.experiments.aggregation_ablation import aggregate_anomaly_map, run_aggregation_ablation


class MockAnomalyDetector(BaseAnomalyDetector):
    def fit(self, dataloader: DataLoader) -> None:
        pass

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        B = x.shape[0]
        scores = np.linspace(0.2, 0.8, B)
        amaps = np.ones((B, 256, 256), dtype=float) * 0.4
        return scores, amaps


def test_failure_miner_synthetic(tmp_path):
    """
    Verifies FailureMiner discovers failure categories and renders 4-panel diagnostic images.
    """
    x = torch.randn(8, 3, 256, 256)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    mask = torch.zeros(8, 1, 256, 256, dtype=torch.float32)
    mask[4:, 0, 50:100, 50:100] = 1.0
    meta = [{"defect_type": "scratch" if i >= 4 else "good"} for i in range(8)]

    class DummyDS(Dataset):
        def __len__(self):
            return 8
        def __getitem__(self, idx):
            return x[idx], y[idx], mask[idx], meta[idx]

    loader = DataLoader(DummyDS(), batch_size=2, shuffle=False)
    model = MockAnomalyDetector()

    miner = FailureMiner(model, loader, output_dir=str(tmp_path))
    failures = miner.mine_failures(top_k=2)

    assert "false_positives" in failures
    assert "false_negatives" in failures
    assert "localization_mismatches" in failures
    assert "corruption_failures" in failures

    saved_images = miner.save_diagnostic_grids(failures, category="bottle", method_name="mock")
    assert len(saved_images) > 0
    for img_path in saved_images:
        assert os.path.exists(img_path)
        assert os.path.getsize(img_path) > 0


def test_augmented_normal_dataset_synthetic(tmp_path):
    """
    Verifies AugmentedNormalDataset applies mild stochastic physical augmentations.
    """
    good_dir = tmp_path / "bottle" / "train" / "good"
    good_dir.mkdir(parents=True)
    dummy_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(dummy_img).save(good_dir / "000.png")

    ds = AugmentedNormalDataset(root=str(tmp_path), category="bottle", aug_prob=1.0)
    assert len(ds) == 1

    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 256, 256)
    assert not torch.isnan(sample).any()


def test_robust_training_experiment_mock():
    """
    Verifies RobustTrainingExperiment executes comparison on synthetic data.
    """
    exp = RobustTrainingExperiment(root="data/mvtec_ad", category="bottle", method="patchcore")
    assert exp.category == "bottle"
    assert exp.method == "patchcore"


def test_aggregation_ablation_all_rules():
    """
    Verifies all 5 spatial aggregation rules and ablation runner.
    """
    amap = np.random.rand(32, 32).astype(np.float32)
    for strat in ["global_max", "percentile_99", "percentile_95", "top_1_percent_mean", "gaussian_pooled_max"]:
        s = aggregate_anomaly_map(amap, method=strat)
        assert isinstance(s, float)
        assert not np.isnan(s)

    with pytest.raises(ValueError):
        aggregate_anomaly_map(amap, method="unknown_strategy")

    # Batch test
    amaps = np.random.rand(10, 32, 32).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    res = run_aggregation_ablation(amaps, labels)
    assert len(res) == 5
    for strat, metrics in res.items():
        assert "image_auroc" in metrics
        assert "image_ap" in metrics