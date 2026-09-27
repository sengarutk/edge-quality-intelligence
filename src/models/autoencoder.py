import os
from typing import Tuple, Optional
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseAnomalyDetector


class _ConvAutoencoderArch(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: (3, 256, 256) -> (128, 16, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: (128, 16, 16) -> (3, 256, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class ConvAutoencoder(BaseAnomalyDetector):
    """
    Convolutional Autoencoder baseline:
    - Trained to reconstruct nominal (defect-free) factory images
    - At inference, defective regions fail to reconstruct accurately
    - Pixel anomaly map = per-pixel Mean Squared Reconstruction Error
    - Image anomaly score = 95th percentile spatial reconstruction error
    """
    def __init__(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ):
        super().__init__(device=device)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = _ConvAutoencoderArch().to(self.device)
        self.criterion = nn.MSELoss()

    def fit(self, dataloader: DataLoader) -> None:
        """
        Fits autoencoder weights on normal training images.
        """
        all_tensors = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            all_tensors.append(x)

        if len(all_tensors) == 0:
            return

        data = torch.cat(all_tensors, dim=0).to(self.device)
        n_samples = data.shape[0]
        batch_size = 16

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i + batch_size]
                x_batch = data[batch_idx]

                optimizer.zero_grad()
                reconstruction = self.model(x_batch)
                loss = self.criterion(reconstruction, x_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes reconstruction residual heatmap and image anomaly scores.
        Returns:
            image_scores: np.ndarray of shape [B]
            anomaly_maps: np.ndarray of shape [B, 256, 256]
        """
        self.model.eval()
        B = x.shape[0]
        x = x.to(self.device)

        with torch.no_grad():
            reconstructed = self.model(x)
            residual = torch.mean((x - reconstructed) ** 2, dim=1).cpu().numpy()

        smoothed_amaps = np.zeros_like(residual)
        image_scores = np.zeros(B, dtype=float)

        for b in range(B):
            smoothed_amaps[b] = scipy.ndimage.gaussian_filter(residual[b], sigma=4)
            image_scores[b] = float(np.percentile(smoothed_amaps[b], 95))

        return image_scores, smoothed_amaps

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state"])
        self.epochs = state.get("epochs", 20)
        self.lr = state.get("lr", 1e-3)
        self.weight_decay = state.get("weight_decay", 1e-5)
