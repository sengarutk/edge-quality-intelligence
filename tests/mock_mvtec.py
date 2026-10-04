"""Tiny random-image dataset in MVTec AD layout, for loader unit tests only (written to tmp dirs)."""

import os

import numpy as np
from PIL import Image


def generate_mock_category(data_root: str, category: str, num_train: int = 5, num_test: int = 4) -> None:
    rng = np.random.default_rng(0)
    dirs = {k: os.path.join(data_root, category, *p) for k, p in {
        "train": ("train", "good"), "good": ("test", "good"), "defect": ("test", "defect"),
        "gt": ("ground_truth", "defect")}.items()}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    img = lambda: Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))  # noqa: E731
    for i in range(num_train):
        img().save(os.path.join(dirs["train"], f"{i:03d}.png"))
    n_good = num_test // 2
    for i in range(n_good):
        img().save(os.path.join(dirs["good"], f"{i:03d}.png"))
    for i in range(num_test - n_good):
        img().save(os.path.join(dirs["defect"], f"{i:03d}.png"))
        mask = np.zeros((256, 256), np.uint8)
        mask[100:150, 100:150] = 255
        Image.fromarray(mask).save(os.path.join(dirs["gt"], f"{i:03d}_mask.png"))
