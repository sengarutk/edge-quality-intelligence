#!/usr/bin/env python3
"""Build per-category PatchCore score banks on MVTec AD for the runtime experiments.

For every category:
  1. Split train/good 80/20 with a fixed seed. PatchCore (ResNet-18, 10% greedy coreset,
     224x224 input, the runtime resolution) is fitted on the 80% part only.
  2. The held-out 20% nominal images give (a) the normalization reference d_ref, the
     99th percentile of their image distances, and (b) the optical blur threshold,
     half of the 5th percentile of their Laplacian variance. No test image is used for
     calibration.
  3. All test images are scored (nominal and defective). Each test nominal image is also
     scored three times with a synthetic specular glare (a blurred bright ellipse), which
     provides real model responses to transient optical artifacts.

Outputs: results/score_bank/<category>.npz and results/score_bank/models/<category>_patchcore.pt
(the model files are large and git-ignored).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_service import focus_measure  # noqa: E402
from src.data.mvtec import IMAGENET_MEAN, IMAGENET_STD, MVTecTest, MVTecTrainNormal  # noqa: E402
from src.metrics.image_metrics import compute_image_auroc  # noqa: E402
from src.models import PatchCore  # noqa: E402
from src.utils import seed_everything  # noqa: E402

IMG = 224


def to_tensor_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (IMG, IMG)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def add_glare(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay a soft saturated ellipse, mimicking a specular reflection."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.float32)
    center = (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h))
    axes = (int(rng.uniform(0.06, 0.14) * w), int(rng.uniform(0.03, 0.08) * h))
    cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, 0.02 * w))[..., None]
    out = img_bgr.astype(np.float32) * (1 - 0.9 * mask) + 255.0 * 0.9 * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def laplacian_var(img_bgr: np.ndarray) -> float:
    return focus_measure(img_bgr, (IMG, IMG))


@torch.no_grad()
def score_images(model: PatchCore, images_bgr: list[np.ndarray], batch: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(images_bgr), batch):
        x = torch.stack([to_tensor_bgr(im) for im in images_bgr[i:i + batch]])
        s, _ = model.predict(x)
        out.append(s)
    return np.concatenate(out)


def build(category: str, data_root: Path, out_dir: Path, seed: int, device: str) -> dict:
    seed_everything(seed)
    train = MVTecTrainNormal(str(data_root), category, img_size=IMG)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_fit = int(round(0.8 * len(idx)))
    fit_idx, cal_idx = sorted(idx[:n_fit].tolist()), sorted(idx[n_fit:].tolist())

    # The memory bank must see the same preprocessing as every query (to_tensor_bgr here and
    # InferenceEngine._preprocess at runtime: cv2 bilinear resize). MVTecTrainNormal resizes with
    # PIL's antialiased filter, which on a ~4x downscale yields visibly different pixels.
    fit_images = torch.stack([to_tensor_bgr(cv2.imread(train.paths[i])) for i in fit_idx])
    model = PatchCore(backbone="resnet18", coreset_sampling_ratio=0.10, device=device, seed=seed)
    model.fit(DataLoader(fit_images, batch_size=16, shuffle=False))
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "models" / f"{category}_patchcore.pt"))

    cal_imgs = [cv2.imread(train.paths[i]) for i in cal_idx]
    cal_d = score_images(model, cal_imgs)
    d_ref = float(np.quantile(cal_d, 0.99))
    lap_cal = np.array([laplacian_var(im) for im in cal_imgs])
    lap_blur = np.array([laplacian_var(cv2.GaussianBlur(im, (0, 0), 3.0)) for im in cal_imgs])
    blur_threshold = float(0.5 * np.quantile(lap_cal, 0.05))

    test = MVTecTest(str(data_root), category, img_size=IMG)
    test_imgs = [cv2.imread(sm[0]) for sm in test.samples]
    labels = np.array([sm[1] for sm in test.samples], dtype=int)
    defect_types = np.array([sm[3]["defect_type"] for sm in test.samples])
    test_d = score_images(model, test_imgs)

    nominal_imgs = [im for im, y in zip(test_imgs, labels) if y == 0]
    glare_imgs = [add_glare(im, rng) for im in nominal_imgs for _ in range(3)]
    glare_d = score_images(model, glare_imgs)
    lap_glare = np.array([laplacian_var(im) for im in glare_imgs])

    frames = np.stack([cv2.resize(im, (IMG, IMG)) for im in cal_imgs[:16]])
    np.savez_compressed(
        out_dir / f"{category}.npz",
        calib_distances=cal_d, test_distances=test_d, test_labels=labels, test_defect_types=defect_types,
        glare_distances=glare_d, d_ref=d_ref, blur_threshold=blur_threshold,
        lap_calib=lap_cal, lap_calib_blurred=lap_blur, lap_glare=lap_glare, frames_bgr=frames,
        test_paths=np.array([sm[0] for sm in test.samples]),
    )
    summary = {
        "category": category,
        "n_fit": len(fit_idx), "n_calib": len(cal_idx),
        "n_test_nominal": int((labels == 0).sum()), "n_test_defect": int((labels == 1).sum()),
        "image_auroc": compute_image_auroc(labels, test_d),
        "d_ref": d_ref,
        "blur_threshold": blur_threshold,
        "frac_calib_flagged_blurry": float(np.mean(lap_cal < blur_threshold)),
        "frac_blurred_detected": float(np.mean(lap_blur < blur_threshold)),
        "median_norm_score_nominal": float(np.median(np.clip(0.5 * test_d[labels == 0] / d_ref, 0, 1))),
        "median_norm_score_defect": float(np.median(np.clip(0.5 * test_d[labels == 1] / d_ref, 0, 1))),
        "median_norm_score_glare": float(np.median(np.clip(0.5 * glare_d / d_ref, 0, 1))),
    }
    print(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "mvtec_ad"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "score_bank"))
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_root, out_dir = Path(args.data_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = args.categories or sorted(os.path.basename(p) for p in glob.glob(str(data_root / "*")) if os.path.isdir(p))
    summaries = [build(c, data_root, out_dir, args.seed, args.device) for c in cats]
    (out_dir / "summary.json").write_text(json.dumps({"seed": args.seed, "input_resolution": IMG,
                                                      "categories": summaries}, indent=2))


if __name__ == "__main__":
    main()
