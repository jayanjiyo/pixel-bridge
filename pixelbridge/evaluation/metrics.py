"""Quality metrics.

OWNER: M3.   Tests: tests/test_evaluation.py
Plan report section 6. All metrics are implemented by us with NumPy.
Inputs are RGB uint8 arrays of the SAME shape unless noted.
"""
from __future__ import annotations

import numpy as np

from ..core.color import rgb_to_gray, rgb_to_lab
from ..core.filters import SOBEL_MAX, filter2d, gaussian_kernel, gradient_magnitude, neighbor


def _check_same_shape(reference: np.ndarray, result: np.ndarray) -> None:
    if reference.shape != result.shape:
        raise ValueError(f"shapes differ: {reference.shape} vs {result.shape}")


def _gray255(img: np.ndarray) -> np.ndarray:
    """Luminance in 0..255 (float64). A (H, W) input is taken as gray already."""
    if img.ndim == 2:
        return img.astype(np.float64)
    return rgb_to_gray(img).astype(np.float64) * 255.0


def _gray01(img: np.ndarray) -> np.ndarray:
    return _gray255(img) / 255.0


def psnr(reference: np.ndarray, result: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB over all channels, peak = 255.
    Returns float('inf') for identical images."""
    _check_same_shape(reference, result)
    diff = reference.astype(np.float64) - result.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(reference: np.ndarray, result: np.ndarray) -> float:
    """Structural similarity (Wang et al. 2004) on the luminance
    (core.color.rgb_to_gray, scaled to 0..255), Gaussian window 11x11 with
    sigma 1.5 (core.filters), C1 = (0.01*255)^2, C2 = (0.03*255)^2.
    Returns the mean SSIM; exactly 1.0 for identical images.

    Local statistics use core.filters.filter2d (replicate border), so the
    SSIM map has the full image size.
    """
    _check_same_shape(reference, result)
    x, y = _gray255(reference), _gray255(result)
    k = gaussian_kernel(11, 1.5)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(a):
        return filter2d(a, k).astype(np.float64)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x * mu_x
    var_y = blur(y * y) - mu_y * mu_y
    cov = blur(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(np.mean(num / den))


def mean_delta_e(reference: np.ndarray, result: np.ndarray) -> float:
    """Mean CIE76 color difference: Euclidean distance in CIELAB
    (core.color.rgb_to_lab), averaged over pixels. 0 for identical images."""
    _check_same_shape(reference, result)
    diff = rgb_to_lab(reference).astype(np.float64) - rgb_to_lab(result).astype(np.float64)
    return float(np.mean(np.sqrt(np.sum(diff * diff, axis=-1))))


def count_isolated_pixels(img: np.ndarray) -> int:
    """Number of pixels whose color differs from all eight neighbors
    (replicated border). Works on RGB (H, W, 3) or index (H, W) images."""
    isolated = np.ones(img.shape[:2], bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            differs = neighbor(img, dy, dx) != img
            if img.ndim == 3:
                differs = differs.any(axis=-1)
            isolated &= differs
    return int(isolated.sum())


def edge_preservation_ratio(reference: np.ndarray, result: np.ndarray,
                            threshold: float = 0.2) -> float:
    """Of the strong edges in `reference` (normalized Sobel magnitude of the
    gray image, gradient_magnitude / core.filters.SOBEL_MAX, > threshold),
    the fraction that are also strong edges in `result`.
    1.0 if reference has no strong edge."""
    _check_same_shape(reference, result)
    ref_edges = gradient_magnitude(_gray01(reference)) / SOBEL_MAX > threshold
    n = int(ref_edges.sum())
    if n == 0:
        return 1.0
    res_edges = gradient_magnitude(_gray01(result)) / SOBEL_MAX > threshold
    return float((ref_edges & res_edges).sum() / n)
