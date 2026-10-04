import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Anchored repository paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "mvtec_ad"

# Official ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transform(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _img_to_tensor(img: Image.Image, img_size: int = 256) -> torch.Tensor:
    t = get_image_transform(img_size)
    return t(img)


def _mask_to_tensor(mask: Image.Image, img_size: int = 256) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    mask_tensor = t(mask)
    return (mask_tensor > 0.5).float()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor [3, H, W] back to an RGB NumPy uint8 image [H, W, 3].
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    arr = denorm.cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).astype(np.uint8)


class MVTecTrainNormal(Dataset):
    """
    Loads normal training images from {root}/{category}/train/good/*.png.
    """
    def __init__(self, root: Optional[str] = None, category: str = "bottle", img_size: int = 256):
        if root is None or str(root) == "":
            root = str(DEFAULT_DATA_DIR)
        elif not os.path.isabs(str(root)):
            candidate = ROOT_DIR / root
            if candidate.exists() or not os.path.exists(str(root)):
                root = str(candidate)

        self.root = root
        self.category = category
        self.img_size = img_size
        self.img_dir = os.path.join(str(root), category, "train", "good")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(
                f"Training directory not found: '{self.img_dir}'.\n"
                f"Install the official MVTec AD archive with:\n"
                f"  python scripts/download_dataset.py --archive /path/to/mvtec_anomaly_detection.tar.xz"
            )

        self.paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")) +
                            glob.glob(os.path.join(self.img_dir, "*.PNG")) +
                            glob.glob(os.path.join(self.img_dir, "*.jpg")))

        if len(self.paths) == 0:
            raise ValueError(f"No training images found in '{self.img_dir}'.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        return _img_to_tensor(img, self.img_size)


class MVTecTest(Dataset):
    """
    Loads test images (good + defects) and corresponding ground truth masks from:
      - {root}/{category}/test/{defect_type}/*.png
      - {root}/{category}/ground_truth/{defect_type}/*_mask.png
    """
    def __init__(self, root: Optional[str] = None, category: str = "bottle", img_size: int = 256):
        if root is None or str(root) == "":
            root = str(DEFAULT_DATA_DIR)
        elif not os.path.isabs(str(root)):
            candidate = ROOT_DIR / root
            if candidate.exists() or not os.path.exists(str(root)):
                root = str(candidate)

        self.root = root
        self.category = category
        self.img_size = img_size
        self.test_dir = os.path.join(str(root), category, "test")
        self.gt_dir = os.path.join(str(root), category, "ground_truth")

        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(
                f"Test directory not found: '{self.test_dir}'.\n"
                f"Install the official MVTec AD archive with:\n"
                f"  python scripts/download_dataset.py --archive /path/to/mvtec_anomaly_detection.tar.xz"
            )

        self.samples: List[Tuple[str, int, Optional[str], Dict[str, Any]]] = []

        defect_types = sorted(os.listdir(self.test_dir))
        for dtype in defect_types:
            dtype_dir = os.path.join(self.test_dir, dtype)
            if not os.path.isdir(dtype_dir):
                continue

            img_paths = sorted(glob.glob(os.path.join(dtype_dir, "*.png")) +
                               glob.glob(os.path.join(dtype_dir, "*.PNG")) +
                               glob.glob(os.path.join(dtype_dir, "*.jpg")))

            is_defect = (dtype != "good")
            y_label = 1 if is_defect else 0

            for ipath in img_paths:
                mask_path = None
                if is_defect:
                    base_name = os.path.splitext(os.path.basename(ipath))[0]
                    cand_mask = os.path.join(self.gt_dir, dtype, f"{base_name}_mask.png")
                    if os.path.exists(cand_mask):
                        mask_path = cand_mask
                    else:
                        mask_path = None

                meta = {
                    "category": category,
                    "defect_type": dtype,
                    "img_path": ipath,
                    "mask_path": mask_path if mask_path is not None else ""
                }
                self.samples.append((ipath, y_label, mask_path, meta))

        if len(self.samples) == 0:
            raise ValueError(f"No test samples found in '{self.test_dir}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, Dict[str, Any]]:
        img_path, label, mask_path, meta = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = _img_to_tensor(img, self.img_size)

        if mask_path is not None and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            mask = _mask_to_tensor(mask_img, self.img_size)
        else:
            mask = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        return x, label, mask, meta
