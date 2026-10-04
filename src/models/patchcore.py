from typing import Optional, Tuple
import os
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm

from .base import BaseAnomalyDetector


class PatchCore(BaseAnomalyDetector):
    """PatchCore-style anomaly detector (Roth et al., CVPR 2022), simplified.

    Patch features come from ResNet layer2 and layer3 (3x3 local average pooling,
    layer3 upsampled to the layer2 grid). The memory bank is a greedy k-center
    coreset computed in a random orthogonal projection (Johnson-Lindenstrauss),
    as in the original method. The image score is the maximum of the
    Gaussian-smoothed (sigma = 4) nearest-neighbor distance map; the original
    paper's neighborhood re-weighting of the image score is not implemented.

    ``coreset_mode="greedy"`` (default) runs exact sequential greedy selection.
    ``coreset_mode="batched"`` adds the ``batch_k`` farthest points per step; it is
    faster but only approximates greedy k-center.
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        coreset_sampling_ratio: float = 0.10,
        projection_dim: int = 128,
        device: Optional[str] = None,
        seed: int = 42,
        coreset_mode: str = "greedy",
        batch_k: int = 50,
    ):
        super().__init__(device=device)
        if coreset_mode not in ("greedy", "batched"):
            raise ValueError("coreset_mode must be 'greedy' or 'batched'")
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.projection_dim = projection_dim
        self.seed = seed
        self.coreset_mode = coreset_mode
        self.batch_k = batch_k
        self._build_backbone(backbone)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.memory_bank: Optional[torch.Tensor] = None

    def _build_backbone(self, backbone: str) -> None:
        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.net = net.to(self.device).eval().requires_grad_(False)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Locally aggregated layer2+layer3 feature map (B, C, H, W)."""
        net = self.net
        stem = net.maxpool(net.relu(net.bn1(net.conv1(x.to(self.device)))))
        layer2 = net.layer2(net.layer1(stem))
        layer3 = net.layer3(layer2)
        pooled2 = self.avg_pool(layer2)
        pooled3 = F.interpolate(self.avg_pool(layer3), size=pooled2.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([pooled2, pooled3], dim=1)

    @torch.no_grad()
    def _extract_multiscale_features(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened patch features (B*H*W, C)."""
        feature_map = self._feature_map(x)
        return feature_map.permute(0, 2, 3, 1).flatten(0, 2)

    def _greedy_coreset_subsampling(self, patches: torch.Tensor) -> torch.Tensor:
        n_patches, dim = patches.shape
        n_select = max(1, int(n_patches * self.coreset_sampling_ratio))
        if n_select >= n_patches:
            return patches

        # Distances are computed in a random orthogonal projection (Johnson-Lindenstrauss);
        # the selected rows are taken from the full-dimensional patches.
        if dim > self.projection_dim:
            g = torch.Generator(device=self.device).manual_seed(self.seed)
            proj = torch.randn(dim, self.projection_dim, device=self.device, generator=g)
            proj, _ = torch.linalg.qr(proj)
            points = patches @ proj
        else:
            points = patches

        g_start = torch.Generator(device="cpu").manual_seed(self.seed)
        start_idx = int(torch.randint(0, n_patches, (1,), generator=g_start).item())
        selected = [start_idx]

        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b; clamp absorbs float cancellation below zero.
        sq_norms = points.pow(2).sum(dim=1)
        min_dist_sq = (sq_norms + sq_norms[start_idx] - 2.0 * points @ points[start_idx]).clamp_(min=0.0)

        step = 1 if self.coreset_mode == "greedy" else self.batch_k
        while len(selected) < n_select:
            k = min(step, n_select - len(selected))
            if k == 1:
                new_idx = torch.argmax(min_dist_sq).view(1)
            else:
                new_idx = torch.topk(min_dist_sq, k=k).indices
            selected.extend(new_idx.tolist())
            queries = points[new_idx]
            dist_sq = (sq_norms[:, None] + sq_norms[new_idx][None, :] - 2.0 * points @ queries.T).clamp_(min=0.0)
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq.min(dim=1).values)

        return patches[selected[:n_select]]

    def fit(self, dataloader: DataLoader) -> None:
        patches = [self._extract_multiscale_features(b[0] if isinstance(b, (list, tuple)) else b) for b in dataloader]
        self.memory_bank = self._greedy_coreset_subsampling(torch.cat(patches, dim=0))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore model is not fitted. Call .fit() first.")

        batch, _, h_in, w_in = x.shape
        feature_map = self._feature_map(x)
        _, channels, h_feat, w_feat = feature_map.shape
        queries = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)

        # Chunked so the (queries x memory bank) distance matrix stays bounded in memory.
        nn_dist = torch.cat([torch.cdist(chunk, self.memory_bank).min(dim=1).values for chunk in queries.split(2048)])
        patch_scores = nn_dist.reshape(batch, 1, h_feat, w_feat)
        anomaly_maps = F.interpolate(patch_scores, size=(h_in, w_in), mode="bilinear", align_corners=False)
        anomaly_maps = anomaly_maps.squeeze(1).cpu().numpy()

        smoothed = np.stack([scipy.ndimage.gaussian_filter(m, sigma=4) for m in anomaly_maps])
        return smoothed.max(axis=(1, 2)).astype(np.float64), smoothed

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "backbone": self.backbone_name,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "projection_dim": self.projection_dim,
            "seed": self.seed,
            "coreset_mode": self.coreset_mode,
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        backbone = state.get("backbone", "resnet18")
        if backbone != self.backbone_name:
            # A resnet50 bank (1536-d patches) cannot be queried with resnet18 features (384-d).
            self._build_backbone(backbone)
        bank = state.get("memory_bank")
        self.memory_bank = bank.to(self.device) if bank is not None else None
        self.coreset_sampling_ratio = state.get("coreset_sampling_ratio", 0.10)
        self.projection_dim = state.get("projection_dim", 128)
        self.seed = state.get("seed", 42)
        self.coreset_mode = state.get("coreset_mode", "greedy")
