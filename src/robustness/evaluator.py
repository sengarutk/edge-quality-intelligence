import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyDetector
from src.metrics.image_metrics import compute_image_auroc, compute_image_ap, compute_optimal_f1
from src.metrics.pixel_metrics import compute_pixel_auroc, compute_pixel_ap, compute_aupro
from src.metrics.calibration import compute_ece
from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error
)
from src.robustness.corruptions import CORRUPTION_TYPES
from src.robustness.dataset import CorruptedMVTecTest


class RobustnessEvaluator:
    """
    Multi-Baseline Robustness Evaluation & Distribution Shift Stress-Testing Engine.
    Computes both:
      1. Signed Mean Performance Change (MPC):
         MPC = 1/18 * sum_{i=1}^{18} (Metric_clean - Metric_corrupted_i)
      2. Non-Negative Mean Robustness Degradation (MRD):
         MRD = 1/18 * sum_{i=1}^{18} max(0, Metric_clean - Metric_corrupted_i)
      3. Operational Robustness under Cost-Calibrated Thresholds (MRD_CWE).
    """
    def __init__(
        self,
        model: BaseAnomalyDetector,
        data_root: str = "data/mvtec_ad",
        category: str = "bottle",
        device: Optional[str] = None,
        root: Optional[str] = None
    ):
        self.model = model
        self.data_root = root if root is not None else data_root
        self.category = category
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def predict_split(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes model inference across entire dataloader split, returning:
        (image_labels, image_scores, ground_truth_masks, pixel_anomaly_maps)
        """
        all_y: List[int] = []
        all_scores: List[float] = []
        all_masks: List[np.ndarray] = []
        all_amaps: List[np.ndarray] = []

        for batch in dataloader:
            x, y, mask, _ = batch
            scores, amaps = self.model.predict(x)

            all_y.extend(y.cpu().numpy().tolist())
            all_scores.extend(scores.tolist())

            m_np = mask.cpu().numpy()
            if m_np.ndim == 4:
                m_np = m_np[:, 0, :, :]
            all_masks.append(m_np)
            all_amaps.append(amaps)

        y_arr = np.array(all_y, dtype=int)
        scores_arr = np.array(all_scores, dtype=float)
        masks_arr = np.concatenate(all_masks, axis=0) if len(all_masks) > 0 else np.zeros((0, 256, 256), dtype=np.float32)
        amaps_arr = np.concatenate(all_amaps, axis=0) if len(all_amaps) > 0 else np.zeros((0, 256, 256), dtype=np.float32)

        return y_arr, scores_arr, masks_arr, amaps_arr

    def evaluate_predictions(
        self,
        y_arr: np.ndarray,
        scores_arr: np.ndarray,
        masks_arr: np.ndarray,
        amaps_arr: np.ndarray
    ) -> Dict[str, float]:
        img_auroc = compute_image_auroc(y_arr, scores_arr)
        img_ap = compute_image_ap(y_arr, scores_arr)
        f1_res = compute_optimal_f1(y_arr, scores_arr)
        pix_auroc = compute_pixel_auroc(masks_arr, amaps_arr)
        pix_ap = compute_pixel_ap(masks_arr, amaps_arr)
        aupro_val = compute_aupro(masks_arr, amaps_arr)
        ece_val = compute_ece(y_arr, scores_arr)

        return {
            "num_samples": float(len(y_arr)),
            "image_auroc": img_auroc,
            "image_ap": img_ap,
            "max_f1": f1_res["max_f1"],
            "optimal_threshold": f1_res["optimal_threshold"],
            "oracle_max_f1": f1_res["oracle_max_f1"],
            "oracle_threshold": f1_res["oracle_threshold"],
            "precision_at_optimal": f1_res["precision_at_optimal"],
            "recall_at_optimal": f1_res["recall_at_optimal"],
            "pixel_auroc": pix_auroc,
            "pixel_ap": pix_ap,
            "aupro": aupro_val,
            "ece": ece_val
        }

    def evaluate_split(self, dataloader: DataLoader) -> Dict[str, float]:
        y_arr, scores_arr, masks_arr, amaps_arr = self.predict_split(dataloader)
        return self.evaluate_predictions(y_arr, scores_arr, masks_arr, amaps_arr)

    def evaluate_operational_robustness(
        self,
        clean_loader: DataLoader,
        corr_loaders: Dict[Tuple[str, int], DataLoader],
        tau: float,
        cost_ratio: float = 10.0,
        prior: float = 0.01
    ) -> Dict[str, Any]:
        """
        Evaluates operational robustness degradation under a fixed decision cutoff (e.g. tau_CCT or tau_budget).
        Computes clean FA@1k, MD@1k, CWE, per-corruption metrics, and Non-Negative Operational Degradation:
          MRD_CWE = 1/18 * sum_{i=1}^{18} max(0, CWE_corrupted_i - CWE_clean)
        """
        clean_y, clean_s, _, _ = self.predict_split(clean_loader)
        clean_fa = compute_fa_at_1k(clean_y, clean_s, tau)
        clean_md = compute_md_at_1k(clean_y, clean_s, tau)
        clean_cwe = compute_cost_weighted_error(clean_y, clean_s, tau, cost_ratio=cost_ratio)

        corrupted_results: List[Dict[str, Any]] = []
        cwe_drops: List[float] = []

        for (ctype, sev), loader in corr_loaders.items():
            corr_y, corr_s, _, _ = self.predict_split(loader)
            corr_fa = compute_fa_at_1k(corr_y, corr_s, tau)
            corr_md = compute_md_at_1k(corr_y, corr_s, tau)
            corr_cwe = compute_cost_weighted_error(corr_y, corr_s, tau, cost_ratio=cost_ratio)

            delta_cwe = corr_cwe - clean_cwe
            delta_fa = corr_fa - clean_fa
            cwe_drops.append(max(0.0, delta_cwe))

            corrupted_results.append({
                "corruption_type": ctype,
                "severity": sev,
                "corrupted_fa_at_1k": corr_fa,
                "corrupted_md_at_1k": corr_md,
                "corrupted_cwe": corr_cwe,
                "delta_cwe": delta_cwe,
                "delta_fa_at_1k": delta_fa
            })

        mrd_cwe = float(np.mean(cwe_drops)) if len(cwe_drops) > 0 else 0.0

        return {
            "clean_fa_at_1k": clean_fa,
            "clean_md_at_1k": clean_md,
            "clean_cwe": clean_cwe,
            "corrupted_results": corrupted_results,
            "mrd_cwe": mrd_cwe
        }

    def run_full_stress_test(
        self,
        severities: List[int] = [1, 2, 3],
        batch_size: int = 4
    ) -> Dict[str, Any]:
        clean_ds = CorruptedMVTecTest(self.data_root, self.category, corruption_type=None)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
        clean_metrics = self.evaluate_split(clean_loader)

        clean_auroc = clean_metrics["image_auroc"]
        clean_aupro = clean_metrics["aupro"]

        corrupted_results: List[Dict[str, Any]] = []
        signed_auroc_drops: List[float] = []
        signed_aupro_drops: List[float] = []
        clipped_auroc_drops: List[float] = []
        clipped_aupro_drops: List[float] = []

        for ctype in CORRUPTION_TYPES:
            for sev in severities:
                corr_ds = CorruptedMVTecTest(
                    self.data_root,
                    self.category,
                    corruption_type=ctype,
                    severity=sev
                )
                corr_loader = DataLoader(corr_ds, batch_size=batch_size, shuffle=False)
                c_metrics = self.evaluate_split(corr_loader)

                delta_auroc = clean_auroc - c_metrics["image_auroc"]
                delta_aupro = clean_aupro - c_metrics["aupro"]

                signed_auroc_drops.append(delta_auroc)
                signed_aupro_drops.append(delta_aupro)
                clipped_auroc_drops.append(max(0.0, delta_auroc))
                clipped_aupro_drops.append(max(0.0, delta_aupro))

                entry = {
                    "corruption_type": ctype,
                    "severity": sev,
                    "image_auroc": c_metrics["image_auroc"],
                    "pixel_auroc": c_metrics["pixel_auroc"],
                    "aupro": c_metrics["aupro"],
                    "delta_image_auroc": delta_auroc,
                    "delta_aupro": delta_aupro,
                    "ece": c_metrics["ece"]
                }
                corrupted_results.append(entry)

        mean_perf_change_auroc = float(np.mean(signed_auroc_drops))
        mean_perf_change_aupro = float(np.mean(signed_aupro_drops))
        non_neg_mrd_auroc = float(np.mean(clipped_auroc_drops))
        non_neg_mrd_aupro = float(np.mean(clipped_aupro_drops))

        return {
            "category": self.category,
            "clean_metrics": clean_metrics,
            "corrupted_results": corrupted_results,
            "mrd_image_auroc": non_neg_mrd_auroc,
            "mrd_aupro": non_neg_mrd_aupro,
            "mean_performance_change_auroc": mean_perf_change_auroc,
            "mean_performance_change_aupro": mean_perf_change_aupro,
            # Unnormalized mean change; not Hendrycks' mCE (which normalizes by a reference model).
        }

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        elif output_path.endswith(".csv"):
            df = pd.DataFrame(results.get("corrupted_results", []))
            df.to_csv(output_path, index=False)