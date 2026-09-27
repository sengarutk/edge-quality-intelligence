import os
import random
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import seed_everything
from src.mvtec import MVTecTrainNormal, _img_to_tensor
from src.models import ConvAutoencoder, PaDiM, PatchCore
from src.robustness.evaluator import RobustnessEvaluator


SEEN_FAMILIES = ("gaussian_blur", "brightness_drop", "jpeg_compression")
HELDOUT_FAMILIES = ("gaussian_noise", "motion_blur", "downscale_restore")


class AugmentedNormalDataset(Dataset):
    """
    Applies mild physical perturbations to nominal training images
    to evaluate whether slight corruption-aware augmentation improves out-of-distribution robustness.
    """
    def __init__(self, root: str, category: str, aug_prob: float = 0.3):
        self.base = MVTecTrainNormal(root, category)
        self.aug_prob = aug_prob

    def __len__(self) -> int:
        return len(self.base)

    def _apply_mild_augmentations(self, img_np: np.ndarray) -> np.ndarray:
        arr = img_np.copy()
        if random.random() < self.aug_prob:
            k = random.choice([3, 5])
            sig = random.uniform(0.5, 1.0)
            arr = cv2.GaussianBlur(arr, (k, k), sigmaX=sig, sigmaY=sig)

        if random.random() < self.aug_prob:
            factor = random.uniform(0.8, 1.1)
            arr = np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if random.random() < self.aug_prob:
            q = random.randint(60, 90)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

        return arr

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.base.paths[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)

        aug_np = self._apply_mild_augmentations(img_np)
        aug_img = Image.fromarray(aug_np)
        return _img_to_tensor(aug_img)


class RobustTrainingExperiment:
    """
    Ablation study: Trains anomaly detectors on standard vs. augmented nominal images,
    evaluating clean accuracy retention vs. corrupted robustness gains (Delta MRD).
    """
    def __init__(
        self,
        root: str,
        category: str,
        method: str = "patchcore",
        device: Optional[str] = None
    ):
        self.root = root
        self.category = category
        self.method = method.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        if self.method == "patchcore":
            return PatchCore(device=self.device, seed=self._seed)
        elif self.method == "padim":
            return PaDiM(device=self.device, random_seed=self._seed)
        elif self.method == "autoencoder":
            return ConvAutoencoder(device=self.device, epochs=20)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def run_comparison(self, seed: int = 42, batch_size: int = 4) -> Dict[str, Any]:
        seed_everything(seed)
        self._seed = seed

        # 1. Train Standard Model on Clean Nominals
        clean_train_ds = MVTecTrainNormal(self.root, self.category)
        clean_loader = DataLoader(clean_train_ds, batch_size=batch_size, shuffle=True)

        model_clean = self._create_model()
        model_clean.fit(clean_loader)

        evaluator_clean = RobustnessEvaluator(model_clean, self.root, self.category, device=self.device)
        clean_stress_results = evaluator_clean.run_full_stress_test(batch_size=batch_size)

        # 2. Train Robust Model on Augmented Nominals
        aug_train_ds = AugmentedNormalDataset(self.root, self.category)
        aug_loader = DataLoader(aug_train_ds, batch_size=batch_size, shuffle=True)

        model_robust = self._create_model()
        model_robust.fit(aug_loader)

        evaluator_robust = RobustnessEvaluator(model_robust, self.root, self.category, device=self.device)
        robust_stress_results = evaluator_robust.run_full_stress_test(batch_size=batch_size)

        # 3. Report degradation separately for corruption families seen during augmentation
        #    (blur, brightness, JPEG) and families never seen in training (noise, motion blur, downscale).
        def split_mrd(results: Dict[str, Any], families) -> float:
            drops = [max(0.0, r["delta_image_auroc"]) for r in results["corrupted_results"]
                     if r["corruption_type"] in families]
            return float(sum(drops) / len(drops)) if drops else 0.0

        def summarize(results: Dict[str, Any]) -> Dict[str, float]:
            return {
                "clean_auroc": results["clean_metrics"]["image_auroc"],
                "clean_aupro": results["clean_metrics"]["aupro"],
                "mrd_image_auroc": results["mrd_image_auroc"],
                "mrd_aupro": results["mrd_aupro"],
                "mrd_auroc_seen_families": split_mrd(results, SEEN_FAMILIES),
                "mrd_auroc_heldout_families": split_mrd(results, HELDOUT_FAMILIES),
            }

        clean_s, robust_s = summarize(clean_stress_results), summarize(robust_stress_results)
        summary = {
            "category": self.category,
            "method": self.method,
            "seed": seed,
            "clean_model": clean_s,
            "robust_model": robust_s,
            "delta_clean_auroc": robust_s["clean_auroc"] - clean_s["clean_auroc"],
            "delta_mrd_auroc": robust_s["mrd_image_auroc"] - clean_s["mrd_image_auroc"],
            "delta_mrd_auroc_heldout": robust_s["mrd_auroc_heldout_families"] - clean_s["mrd_auroc_heldout_families"],
        }
        return summary
