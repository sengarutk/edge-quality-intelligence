"""Optical Evidence Artifact & Overlay Generator.

Manages saving and loading of visual inspection evidence with blended defect heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from loguru import logger


class EvidenceManager:
    """Artifact manager for visual inspection defect overlays and optical records."""

    def __init__(self, storage_dir: str = "data/evidence") -> None:
        """Initialize EvidenceManager and ensure storage directory exists.

        Args:
            storage_dir: Filesystem path to evidence image archive.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized EvidenceManager at {self.storage_dir.resolve()}")

    def save_evidence(
        self,
        frame: np.ndarray,
        heatmap: Optional[np.ndarray],
        frame_id: str,
    ) -> str:
        """Create and save a composite evidence image with anomaly heatmap overlay.

        Args:
            frame: Raw input camera image (uint8).
            heatmap: Optional 2D anomaly heatmap (float32 [0.0, 1.0]).
            frame_id: Frame UUID used for naming.

        Returns:
            Relative filesystem path to saved PNG evidence artifact.
        """
        # Ensure frame is 3-channel uint8
        if frame.dtype != np.uint8:
            is_unit = np.issubdtype(frame.dtype, np.floating) and float(frame.max()) <= 1.01
            frame_norm = np.clip(frame * 255.0 if is_unit else frame, 0, 255).astype(np.uint8)
        else:
            frame_norm = frame.copy()

        if len(frame_norm.shape) == 2:
            frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
        elif frame_norm.shape[2] == 1:
            frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame_norm.copy()

        h, w = frame_bgr.shape[:2]

        if heatmap is not None:
            # Resize heatmap to match frame resolution
            if heatmap.shape[:2] != (h, w):
                hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                hm_resized = heatmap

            # Normalize to 0-255 uint8
            hm_uint8 = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)
            color_hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

            # Alpha blend: 0.6 original + 0.4 heatmap overlay
            blended = cv2.addWeighted(frame_bgr, 0.6, color_hm, 0.4, 0)

            # Horizontal side-by-side composite
            composite = np.hstack([frame_bgr, blended])
        else:
            # Optical diagnostic badge on raw frame
            composite = frame_bgr.copy()
            cv2.rectangle(composite, (10, 10), (190, 42), (20, 20, 20), -1)
            cv2.putText(
                composite,
                "RAW EVIDENCE",
                (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        safe_id = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(frame_id))[:128]
        if not safe_id:
            raise ValueError("frame_id must contain at least one safe character")
        out_path = self.storage_dir / f"{safe_id}.png"
        if not cv2.imwrite(str(out_path), composite):
            raise IOError(f"Failed to write evidence image {out_path}")
        logger.debug(f"Saved optical evidence artifact to {out_path}")
        return str(out_path)

    def load_evidence(self, uri: str) -> Optional[np.ndarray]:
        """Safely load evidence image from local disk.

        Args:
            uri: Filesystem path to evidence image.

        Returns:
            np.ndarray image in BGR format or None if file not found.
        """
        path = Path(uri)
        if not path.is_file():
            logger.warning(f"Evidence artifact not found at {uri}")
            return None
        img = cv2.imread(str(path))
        return img