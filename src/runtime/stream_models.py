"""Non-IID temporal stream models and cyber-physical failure generators.

Provides block-correlated Markov state transitions, non-stationary thermal drift,
and Poisson-clustered micro-fracture defect arrival sequences for robust edge benchmarking.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class MarkovianDefectGenerator:
    """Two-state discrete-time Markov chain for temporally correlated defect sequences.

    State 0: NOMINAL
    State 1: DEFECTIVE
    """

    def __init__(
        self,
        p_defect_given_defect: float = 0.85,
        p_nominal_given_nominal: float = 0.95,
        seed: int = 42,
    ) -> None:
        """Initialize Markovian defect generator.

        Args:
            p_defect_given_defect: Persistence probability of staying in defect state.
            p_nominal_given_nominal: Persistence probability of staying in nominal state.
            seed: Deterministic random seed.
        """
        self.p_dd = float(p_defect_given_defect)
        self.p_nn = float(p_nominal_given_nominal)
        self.rng = np.random.RandomState(seed)
        self.current_state: int = 0  # 0: Nominal, 1: Defect

    def step(self) -> bool:
        """Advance Markov chain by one temporal step.

        Returns:
            True if current step is defective, False if nominal.
        """
        r = self.rng.uniform(0.0, 1.0)
        if self.current_state == 1:
            self.current_state = 1 if r < self.p_dd else 0
        else:
            self.current_state = 0 if r < self.p_nn else 1

        return bool(self.current_state == 1)

    def generate_sequence(self, n_steps: int) -> List[bool]:
        """Generate a block-correlated sequence of defect indicators."""
        return [self.step() for _ in range(n_steps)]


class ThermalDriftProfile:
    """Non-stationary temperature drift model simulating factory diurnal heating cycles."""

    def __init__(
        self,
        base_temp_c: float = 45.0,
        peak_temp_c: float = 85.0,
        drift_type: str = "linear",  # "linear" | "exponential" | "sinusoidal"
        ramp_steps: int = 300,
    ) -> None:
        """Initialize thermal drift profile.

        Args:
            base_temp_c: Initial ambient operating temperature in Celsius.
            peak_temp_c: Peak operating temperature in Celsius.
            drift_type: Type of temperature profile.
            ramp_steps: Duration over which transition occurs.
        """
        self.base_temp_c = float(base_temp_c)
        self.peak_temp_c = float(peak_temp_c)
        self.drift_type = drift_type.lower()
        self.ramp_steps = max(int(ramp_steps), 1)

    def get_temperature(self, step: int, noise_std: float = 0.5, seed: Optional[int] = None) -> float:
        """Compute instantaneous temperature at given step.

        Args:
            step: Zero-indexed temporal cycle.
            noise_std: Additive Gaussian sensor noise std dev.
            seed: Optional seed for noise generation.

        Returns:
            Instantaneous temperature in °C.
        """
        rng = np.random.RandomState(seed if seed is not None else step)
        norm_t = min(max(step / self.ramp_steps, 0.0), 1.0)

        if self.drift_type == "exponential":
            factor = (np.exp(3.0 * norm_t) - 1.0) / (np.exp(3.0) - 1.0)
        elif self.drift_type == "sinusoidal":
            factor = 0.5 * (1.0 - np.cos(np.pi * norm_t))
        else:  # default linear
            factor = norm_t

        mean_temp = self.base_temp_c + factor * (self.peak_temp_c - self.base_temp_c)
        noise = rng.normal(0.0, noise_std)
        return float(mean_temp + noise)


class PoissonBurstDefectGenerator:
    """Poisson-distributed clustered arrival model for micro-fractures and surface defects."""

    def __init__(
        self,
        mean_cluster_rate: float = 0.02,
        mean_cluster_length: int = 5,
        seed: int = 42,
    ) -> None:
        """Initialize Poisson burst defect generator.

        Args:
            mean_cluster_rate: Poisson rate parameter lambda for cluster initiation.
            mean_cluster_length: Expected duration in steps of each defect burst.
            seed: Deterministic random seed.
        """
        self.rate = float(mean_cluster_rate)
        self.cluster_len = int(mean_cluster_length)
        self.rng = np.random.RandomState(seed)
        self.remaining_burst_steps: int = 0

    def step(self) -> bool:
        """Advance one temporal step.

        Returns:
            True if current step falls within an active defect burst.
        """
        if self.remaining_burst_steps > 0:
            self.remaining_burst_steps -= 1
            return True

        # Check if a new cluster initiates
        num_new = self.rng.poisson(self.rate)
        if num_new > 0:
            burst_duration = max(1, int(self.rng.poisson(self.cluster_len)))
            self.remaining_burst_steps = burst_duration - 1
            return True

        return False

    def generate_sequence(self, n_steps: int) -> List[bool]:
        """Generate sequence of Poisson burst defect indicators."""
        return [self.step() for _ in range(n_steps)]


class MixedCorruptionStream:
    """Simulates mixed optical, electrical, and compression noise on visual edge streams."""

    def __init__(
        self,
        p_corrupt: float = 0.25,
        noise_sigma: float = 25.0,
        blur_kernel: int = 15,
        jpeg_quality: int = 25,
        seed: int = 42,
    ) -> None:
        """Initialize mixed corruption generator.

        Args:
            p_corrupt: Probability that a given frame is subjected to corruptions.
            noise_sigma: Standard deviation of additive Gaussian noise.
            blur_kernel: Kernel diameter for Gaussian optical defocus blur.
            jpeg_quality: Quality factor (1-100) for lossy compression artifacts.
            seed: Random seed for deterministic reproducibility.
        """
        self.p_corrupt = float(np.clip(p_corrupt, 0.0, 1.0))
        self.noise_sigma = float(noise_sigma)
        self.blur_k = blur_kernel if (blur_kernel % 2 == 1) else blur_kernel + 1
        self.jpeg_q = int(np.clip(jpeg_quality, 5, 95))
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def corrupt_frame(
        self,
        frame: np.ndarray,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply stochastic corruptions to an image frame.

        Args:
            frame: Input uint8 RGB image array (H, W, 3).
            step: Optional cycle index for deterministic variation.

        Returns:
            Tuple of (corrupted_frame, list_of_applied_corruptions).
        """
        if self.p_corrupt <= 0.0 or frame is None:
            return frame, []

        # Seeded by (seed, step) so that different seeds give different corruption sequences.
        local_rng = np.random.RandomState((self.seed * 1_000_003 + step) % (2**32)) if step is not None else self.rng
        if local_rng.uniform(0.0, 1.0) > self.p_corrupt:
            return frame.copy(), []

        # Each corruption fires independently; a frame selected for corruption gets at least one.
        chosen = [local_rng.uniform() < 0.6, local_rng.uniform() < 0.5, local_rng.uniform() < 0.5]
        if not any(chosen):
            chosen[int(local_rng.randint(3))] = True
        applied: List[str] = []
        result_frame = frame.copy()

        if chosen[0]:  # sensor / RF noise
            noise = local_rng.normal(0.0, self.noise_sigma, size=frame.shape)
            result_frame = np.clip(result_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            applied.append("GAUSSIAN_NOISE")
        if chosen[1]:  # defocus blur
            result_frame = cv2.GaussianBlur(result_frame, (self.blur_k, self.blur_k), 0)
            applied.append("OPTICAL_BLUR")
        if chosen[2]:  # lossy compression
            ok, enc = cv2.imencode(".jpg", result_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if ok:
                flag = cv2.IMREAD_COLOR if result_frame.ndim == 3 else cv2.IMREAD_GRAYSCALE
                result_frame = cv2.imdecode(enc, flag)
                applied.append("JPEG_COMPRESSION")

        return result_frame, applied
