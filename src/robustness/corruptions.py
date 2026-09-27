import cv2
import numpy as np
from PIL import Image
from typing import Optional, Union

CORRUPTION_TYPES = [
    "gaussian_blur",
    "motion_blur",
    "brightness_drop",
    "gaussian_noise",
    "jpeg_compression",
    "downscale_restore"
]


def _to_numpy_rgb(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            arr = np.clip(img, 0, 255).astype(np.uint8)
        else:
            arr = img.copy()
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(img)}")
    return arr


def apply_gaussian_blur(img: Union[Image.Image, np.ndarray], severity: int = 1) -> np.ndarray:
    """
    Simulates optical defocus and lens softening.
    """
    arr = _to_numpy_rgb(img)
    params = {
        1: ((5, 5), 1.0),
        2: ((9, 9), 2.0),
        3: ((15, 15), 3.5),
    }
    ksize, sigma = params.get(severity, ((5, 5), 1.0))
    blurred = cv2.GaussianBlur(arr, ksize, sigmaX=sigma, sigmaY=sigma)
    return blurred.astype(np.uint8)


def apply_motion_blur(img: Union[Image.Image, np.ndarray], severity: int = 1) -> np.ndarray:
    """
    Simulates high-speed conveyor vibration and line-scan camera motion blur.
    """
    arr = _to_numpy_rgb(img)
    kernel_sizes = {1: 5, 2: 11, 3: 19}
    ksize = kernel_sizes.get(severity, 5)

    # Diagonal motion kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    np.fill_diagonal(kernel, 1.0)
    kernel = kernel / ksize

    blurred = cv2.filter2D(arr, -1, kernel)
    return blurred.astype(np.uint8)


def apply_brightness_drop(img: Union[Image.Image, np.ndarray], severity: int = 1) -> np.ndarray:
    """
    Simulates decaying factory LED lighting, shadow angles, and exposure drops.
    """
    arr = _to_numpy_rgb(img)
    factors = {1: 0.75, 2: 0.50, 3: 0.30}
    factor = factors.get(severity, 0.75)
    dropped = arr.astype(np.float32) * factor
    return np.clip(dropped, 0, 255).astype(np.uint8)


def apply_gaussian_noise(
    img: Union[Image.Image, np.ndarray], severity: int = 1, seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulates high-ISO sensor noise in low-light factory environments.
    """
    arr = _to_numpy_rgb(img)
    sigmas = {1: 15.0, 2: 30.0, 3: 50.0}
    sigma = sigmas.get(severity, 15.0)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, arr.shape)
    noisy = arr.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img: Union[Image.Image, np.ndarray], severity: int = 1) -> np.ndarray:
    """
    Simulates bandwidth-constrained edge network streaming and compression artifacts.
    """
    arr = _to_numpy_rgb(img)
    qualities = {1: 50, 2: 25, 3: 10}
    quality = qualities.get(severity, 50)

    # RGB to BGR for cv2 imencode
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        return arr

    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return decoded_rgb.astype(np.uint8)


def apply_downscale_restore(img: Union[Image.Image, np.ndarray], severity: int = 1) -> np.ndarray:
    """
    Simulates low-cost edge camera sensor downscaling and bilinear interpolation blur.
    """
    arr = _to_numpy_rgb(img)
    h, w = arr.shape[:2]
    down_scales = {1: 128, 2: 64, 3: 32}
    target_dim = down_scales.get(severity, 128)

    downscaled = cv2.resize(arr, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored.astype(np.uint8)


def apply_corruption(
    img: Union[Image.Image, np.ndarray],
    corruption_type: str,
    severity: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Routes corruption_type to the corresponding physical degradation generator.
    """
    dispatch = {
        "gaussian_blur": apply_gaussian_blur,
        "motion_blur": apply_motion_blur,
        "brightness_drop": apply_brightness_drop,
        "gaussian_noise": apply_gaussian_noise,
        "jpeg_compression": apply_jpeg_compression,
        "downscale_restore": apply_downscale_restore,
    }

    if corruption_type not in dispatch:
        raise ValueError(
            f"Unknown corruption type: '{corruption_type}'. Supported: {CORRUPTION_TYPES}"
        )

    if severity not in [1, 2, 3]:
        raise ValueError(f"Severity must be in [1, 2, 3], got {severity}")

    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(img, severity=severity, seed=seed)
    return dispatch[corruption_type](img, severity=severity)
