"""Quality metrics.

OWNER: M3.   Tests: tests/test_evaluation.py
Plan report section 6. All metrics are implemented by us with NumPy.
Inputs are RGB uint8 arrays of the SAME shape unless noted.
"""
from __future__ import annotations

import numpy as np


def psnr(reference: np.ndarray, result: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB over all channels, peak = 255.
    Returns float('inf') for identical images."""
    raise NotImplementedError("TODO(M3): psnr")


def ssim(reference: np.ndarray, result: np.ndarray) -> float:
    """Structural similarity (Wang et al. 2004) on the luminance
    (core.color.rgb_to_gray, scaled to 0..255), Gaussian window 11x11 with
    sigma 1.5 (core.filters), C1 = (0.01*255)^2, C2 = (0.03*255)^2.
    Returns the mean SSIM; exactly 1.0 for identical images."""
    raise NotImplementedError("TODO(M3): ssim")


def mean_delta_e(reference: np.ndarray, result: np.ndarray) -> float:
    """Mean CIE76 color difference: Euclidean distance in CIELAB
    (core.color.rgb_to_lab), averaged over pixels. 0 for identical images."""
    raise NotImplementedError("TODO(M3): mean_delta_e")


def count_isolated_pixels(img: np.ndarray) -> int:
    """Number of pixels whose color differs from all eight neighbors
    (replicated border). Works on RGB (H, W, 3) or index (H, W) images."""
    raise NotImplementedError("TODO(M3): count_isolated_pixels")


def edge_preservation_ratio(reference: np.ndarray, result: np.ndarray,
                            threshold: float = 0.2) -> float:
    """Of the strong edges in `reference` (normalized Sobel magnitude of the
    gray image, gradient_magnitude / core.filters.SOBEL_MAX, > threshold),
    the fraction that are also strong edges in `result`.
    1.0 if reference has no strong edge."""
    raise NotImplementedError("TODO(M3): edge_preservation_ratio")
