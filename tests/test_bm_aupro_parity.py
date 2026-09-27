import numpy as np

from src.metrics.pixel_metrics import compute_aupro
from src.metrics.reference_aupro import compute_aupro_reference


def _masks_and_maps(seed: int = 42):
    rng = np.random.RandomState(seed)
    masks = np.zeros((10, 64, 64), dtype=np.uint8)
    for i in range(10):
        if i % 2 == 0:
            masks[i, 10:20, 10:20] = 1
            masks[i, 40:50, 40:50] = 1
        elif i % 3 == 0:
            masks[i, 25:35, 25:35] = 1
    amaps = rng.rand(10, 64, 64)
    amaps[masks == 1] += 0.4
    return masks, amaps


def test_grid_aupro_matches_exact_reference():
    """The 200-threshold grid approximation must stay close to the exact step curve."""
    masks, amaps = _masks_and_maps()
    assert abs(compute_aupro(masks, amaps, num_thresholds=200) - compute_aupro_reference(masks, amaps)) < 2e-3


def test_reference_perfect_separation_is_one():
    masks, _ = _masks_and_maps()
    amaps = masks.astype(float) + 0.01 * np.random.RandomState(0).rand(*masks.shape)
    assert np.isclose(compute_aupro_reference(masks, amaps), 1.0, atol=1e-6)


def test_reference_random_scores_near_chance():
    """For scores independent of the masks, PRO(f) ~ f, so AU-PRO(0.3) ~ 0.15."""
    masks, _ = _masks_and_maps()
    amaps = np.random.RandomState(1).rand(*masks.shape)
    assert abs(compute_aupro_reference(masks, amaps) - 0.15) < 0.03


def test_aupro_edge_cases():
    masks = np.zeros((5, 32, 32), dtype=np.uint8)
    amaps = np.ones((5, 32, 32), dtype=np.float64)
    assert compute_aupro(masks, amaps) == 0.0
    assert compute_aupro_reference(masks, amaps) == 0.0
