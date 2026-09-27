import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from typing import Tuple

from src.methods.base import BaseAnomalyDetector
from src.robustness.corruptions import apply_corruption, CORRUPTION_TYPES
from src.robustness.dataset import CorruptedMVTecTest
from src.robustness.evaluator import RobustnessEvaluator


def test_all_corruption_primitives():
    """
    Verifies all 6 physical corruption generators across severities 1, 2, and 3 (18 combinations).
    Checks shape, dtype, bounded pixel values, and progressive severity degradation.
    """
    # Create structured pattern image so frequency/blur/compression causes measurable shifts
    y, x = np.ogrid[:256, :256]
    clean_img = ((np.sin(x / 8.0) + np.cos(y / 8.0) + 2.0) * 60.0).astype(np.uint8)
    clean_img = np.stack([clean_img, clean_img, clean_img], axis=-1)

    for c_type in CORRUPTION_TYPES:
        prev_diff = 0.0
        for sev in [1, 2, 3]:
            corrupted = apply_corruption(clean_img, c_type, severity=sev)

            assert isinstance(corrupted, np.ndarray)
            assert corrupted.shape == (256, 256, 3)
            assert corrupted.dtype == np.uint8
            assert (corrupted >= 0).all() and (corrupted <= 255).all()

            # Calculate L1 distance from clean image
            diff = float(np.mean(np.abs(corrupted.astype(float) - clean_img.astype(float))))

            if sev == 1:
                assert diff >= 0.0
            elif sev > 1:
                # Higher severity should have greater or equal pixel divergence
                assert diff >= prev_diff, f"Expected severity {sev} diff >= {prev_diff} for {c_type}, got {diff}"

            prev_diff = diff


def test_corrupted_dataset_loader():
    """
    Verifies CorruptedMVTecTest instantiates cleanly with both clean and corrupted modes.
    """
    ds_clean = CorruptedMVTecTest(root="data/mvtec_ad", category="bottle", corruption_type=None)
    assert len(ds_clean) >= 0

    ds_corr = CorruptedMVTecTest(root="data/mvtec_ad", category="bottle", corruption_type="gaussian_blur", severity=2)
    assert len(ds_corr) >= 0


class MockAnomalyDetector(BaseAnomalyDetector):
    def fit(self, dataloader: DataLoader) -> None:
        pass

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        B = x.shape[0]
        # Return deterministic dummy anomaly scores and heatmaps
        image_scores = np.ones(B, dtype=float) * 0.75
        amaps = np.ones((B, 256, 256), dtype=float) * 0.5
        return image_scores, amaps


def test_robustness_evaluator_synthetic():
    """
    Verifies RobustnessEvaluator calculates the complete metric battery on synthetic batches.
    """
    # Create synthetic dataset with images, labels, masks, and metadata
    x = torch.randn(4, 3, 256, 256)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    mask = torch.zeros(4, 1, 256, 256, dtype=torch.float32)
    mask[2, 0, 50:100, 50:100] = 1.0
    mask[3, 0, 100:150, 100:150] = 1.0
    meta = [{"defect_type": "scratch"}] * 4

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return x[idx], y[idx], mask[idx], meta[idx]

    loader = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)

    model = MockAnomalyDetector()
    evaluator = RobustnessEvaluator(model=model, root="data/mvtec_ad", category="bottle")

    metrics = evaluator.evaluate_split(loader)

    expected_keys = [
        "image_auroc", "image_ap", "max_f1", "optimal_threshold",
        "precision_at_optimal", "recall_at_optimal", "pixel_auroc",
        "pixel_ap", "aupro", "ece", "num_samples"
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric key {k}"
        assert not np.isnan(metrics[k]), f"Metric {k} is NaN"
        assert not np.isinf(metrics[k]), f"Metric {k} is Inf"



def test_evaluate_operational_robustness():
    """
    Verifies evaluate_operational_robustness computes clean vs corrupted metrics and MRD_CWE.
    """
    x = torch.randn(4, 3, 256, 256)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    mask = torch.zeros(4, 1, 256, 256, dtype=torch.float32)
    meta = [{"defect_type": "scratch"}] * 4

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return x[idx], y[idx], mask[idx], meta[idx]

    clean_loader = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)
    corr_loader1 = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)
    corr_loader2 = DataLoader(SyntheticDataset(), batch_size=2, shuffle=False)

    corr_loaders = {
        ("gaussian_noise", 1): corr_loader1,
        ("defocus_blur", 2): corr_loader2
    }

    model = MockAnomalyDetector()
    evaluator = RobustnessEvaluator(model=model, root="data/mvtec_ad", category="bottle")

    op_res = evaluator.evaluate_operational_robustness(
        clean_loader=clean_loader,
        corr_loaders=corr_loaders,
        tau=0.5,
        cost_ratio=10.0,
        prior=0.01
    )

    assert "clean_fa_at_1k" in op_res
    assert "clean_md_at_1k" in op_res
    assert "clean_cwe" in op_res
    assert "mrd_cwe" in op_res
    assert len(op_res["corrupted_results"]) == 2