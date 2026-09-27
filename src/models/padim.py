from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from scipy.ndimage import gaussian_filter

from .base import BaseAnomalyDetector


class PaDiM(BaseAnomalyDetector):
    """
    PaDiM-inspired Anomaly Detector.
    Models normal patch distributions as localized multivariate Gaussian distributions with
    Fixed Diagonal Covariance Regularization (Sigma + lambda * I, lambda=0.01).
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        d_reduced: int = 100,
        device: Optional[str] = None,
        sigma: float = 4.0,
        random_seed: int = 42,
        d_dim: Optional[int] = None,
        seed: Optional[int] = None
    ):
        super().__init__(device=device)
        self.d_reduced = d_dim if d_dim is not None else d_reduced
        self.sigma = sigma
        self.random_seed = seed if seed is not None else random_seed

        # Load ResNet-18
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.feature_extractor = models.resnet18(weights=weights).to(self.device)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mean_grid: Optional[torch.Tensor] = None
        self.inv_cov_grid: Optional[torch.Tensor] = None
        self.idx_selected: Optional[torch.Tensor] = None

    @property
    def mean(self) -> Optional[torch.Tensor]:
        return self.mean_grid

    @property
    def cov(self) -> Optional[torch.Tensor]:
        """Inverse is stored; the covariance itself is recovered on demand."""
        return None if self.inv_cov_grid is None else torch.linalg.inv(self.inv_cov_grid)

    @property
    def inv_cov(self) -> Optional[torch.Tensor]:
        return self.inv_cov_grid

    def _extract_multiscale_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f_maps: List[torch.Tensor] = []

        feat = self.feature_extractor.conv1(x)
        feat = self.feature_extractor.bn1(feat)
        feat = self.feature_extractor.relu(feat)
        feat = self.feature_extractor.maxpool(feat)

        feat1 = self.feature_extractor.layer1(feat)
        feat2 = self.feature_extractor.layer2(feat1)
        feat3 = self.feature_extractor.layer3(feat2)

        f_maps = [feat1, feat2, feat3]
        target_size = feat1.shape[-2:]

        resized_fmaps = []
        for f in f_maps:
            if f.shape[-2:] != target_size:
                resized = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            else:
                resized = f
            resized_fmaps.append(resized)

        cat_fmaps = torch.cat(resized_fmaps, dim=1)
        return cat_fmaps

    def fit(self, dataloader: DataLoader) -> None:
        all_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                feats = self._extract_multiscale_features(x)
                all_embeddings.append(feats)

        embeddings = torch.cat(all_embeddings, dim=0)
        B, C, H, W = embeddings.shape

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.random_seed)
        if self.d_reduced < C:
            self.idx_selected = torch.randperm(C, generator=gen)[:self.d_reduced].to(self.device)
            embeddings = torch.index_select(embeddings, 1, self.idx_selected)
        else:
            self.idx_selected = torch.arange(C, device=self.device)

        _, d, H, W = embeddings.shape
        L = H * W
        feats_flat = embeddings.permute(0, 2, 3, 1).reshape(B, L, d)

        mean_grid = torch.mean(feats_flat, dim=0)

        # Fixed diagonal Tikhonov regularization for numerical invertibility: Sigma + lambda * I (lambda = 0.01)
        cov_grid = torch.zeros((L, d, d), device=self.device)
        for p in range(L):
            diff = feats_flat[:, p, :] - mean_grid[p, :]
            cov = torch.mm(diff.t(), diff) / (B - 1)
            cov_grid[p] = cov + 0.01 * torch.eye(d, device=self.device)

        inv_cov_grid = torch.linalg.inv(cov_grid)

        # Shape mean_grid as (d, H, W) for standard evaluation compatibility
        self.mean_grid = mean_grid.reshape(H, W, d).permute(2, 0, 1)
        self.inv_cov_grid = inv_cov_grid.reshape(H, W, d, d)

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean_grid is None or self.inv_cov_grid is None:
            raise RuntimeError("PaDiM model is not fitted yet.")

        x = x.to(self.device)
        with torch.no_grad():
            feats = self._extract_multiscale_features(x)
            feats = torch.index_select(feats, 1, self.idx_selected)
            B, d, H, W = feats.shape

            # diff: (B, d, H, W) -> permute to (B, H, W, d)
            diff = (feats - self.mean_grid.unsqueeze(0)).permute(0, 2, 3, 1)

            # Vectorized Mahalanobis distance: (B, H, W, d) x (H, W, d, d) x (B, H, W, d) -> (B, H, W)
            dist_sq = torch.einsum("bhwd,hwde,bhwe->bhw", diff, self.inv_cov_grid, diff)
            dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))

            dist_maps = dist.unsqueeze(1)
            dist_maps = F.interpolate(dist_maps, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
            dist_maps = dist_maps.squeeze(1).cpu().numpy()

        smoothed_amaps = []
        for i in range(B):
            amap = gaussian_filter(dist_maps[i], sigma=self.sigma)
            smoothed_amaps.append(amap)

        amaps_arr = np.stack(smoothed_amaps, axis=0)
        image_scores = np.max(amaps_arr, axis=(1, 2))

        return image_scores, amaps_arr

    def save(self, path: str) -> None:
        torch.save({
            "mean_grid": self.mean_grid,
            "inv_cov_grid": self.inv_cov_grid,
            "idx_selected": self.idx_selected,
            "d_reduced": self.d_reduced,
            "sigma": self.sigma,
            "random_seed": self.random_seed
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.mean_grid = checkpoint["mean_grid"].to(self.device)
        self.inv_cov_grid = checkpoint["inv_cov_grid"].to(self.device)
        self.idx_selected = checkpoint["idx_selected"].to(self.device)
        self.d_reduced = checkpoint["d_reduced"]
        self.sigma = checkpoint["sigma"]
        self.random_seed = checkpoint["random_seed"]
