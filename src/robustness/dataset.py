import os
import zlib
from typing import Optional, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.mvtec import _img_to_tensor
from .corruptions import apply_corruption, CORRUPTION_TYPES


class CorruptedMVTecTest(Dataset):
    """
    Corrupted MVTec AD test dataset wrapper.
    Applies physical distribution shifts to test images prior to tensor conversion & ImageNet normalization.
    Ground truth masks remain untouched.
    """
    def __init__(
        self,
        root: str,
        category: str,
        corruption_type: Optional[str] = None,
        severity: int = 1
    ):
        self.root = root
        self.category = category
        self.corruption_type = corruption_type
        self.severity = severity

        self.test_dir = os.path.join(root, category, "test")
        self.gt_dir = os.path.join(root, category, "ground_truth")

        self.samples = []
        if os.path.exists(self.test_dir):
            defect_types = sorted(os.listdir(self.test_dir))
            for d in defect_types:
                ddir = os.path.join(self.test_dir, d)
                if not os.path.isdir(ddir):
                    continue
                for p in sorted(os.listdir(ddir)):
                    if not p.endswith(".png"):
                        continue
                    img_path = os.path.join(ddir, p)
                    y = 0 if d == "good" else 1

                    mask_path = None
                    if y == 1:
                        base = p.replace(".png", "")
                        mask_path = os.path.join(self.gt_dir, d, base + "_mask.png")

                    self.samples.append((img_path, y, mask_path, d))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, Dict[str, str]]:
        img_path, y, mask_path, defect_type = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Apply image degradation if specified
        if self.corruption_type is not None:
            # Per-image seed: the same image always receives the same noise realization.
            seed = zlib.crc32(f"{img_path}|{self.corruption_type}|{self.severity}".encode()) & 0xFFFFFFFF
            corrupted_np = apply_corruption(img, self.corruption_type, severity=self.severity, seed=seed)
            img = Image.fromarray(corrupted_np)

        # Standardize RGB image with ImageNet mean/std
        x = _img_to_tensor(img)

        # Ground truth masks remain unstandardized binary float tensors
        # Nearest-neighbor resize keeps masks identical to the clean MVTecTest loader.
        mask_tfm = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        if mask_path is not None and os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")
            mask = (mask_tfm(m) > 0.5).float()
        else:
            mask = torch.zeros((1, 256, 256), dtype=torch.float32)

        meta = {
            "path": img_path,
            "defect_type": defect_type,
            "corruption_type": self.corruption_type or "clean",
            "severity": str(self.severity) if self.corruption_type else "0"
        }
        return x, int(y), mask, meta
