"""Module A step 1: preprocessing.

OWNER: M2.   Tests: tests/test_module_a_front.py
Plan report section 4.2.
"""
from __future__ import annotations

import numpy as np

from ..core import filters, io, resize


def preprocess(photo: np.ndarray, working_size: int = 512, smooth: bool = True,
               enhance_amount: float = 0.0) -> np.ndarray:
    """Resize so that the longer side is at most `working_size` (area
    averaging), then optionally smooth and enhance. Implemented in the
    skeleton; it only calls the functions below.
    """
    h, w = photo.shape[:2]
    longer = max(h, w)
    if longer > working_size:
        s = working_size / longer
        photo = resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    if smooth:
        photo = bilateral_filter(photo)
    if enhance_amount > 0:
        photo = enhance(photo, enhance_amount)
    return photo


def bilateral_filter(img: np.ndarray, diameter: int = 7, sigma_color: float = 0.1,
                     sigma_space: float = 3.0) -> np.ndarray:
    """Edge-preserving smoothing (Tomasi and Manduchi 1998).

    img         : (H, W, 3) uint8
    diameter    : window size (odd)
    sigma_color : range sigma on float colors in [0, 1] (Euclidean RGB distance)
    sigma_space : spatial sigma in pixels
    returns     : (H, W, 3) uint8

    Speed: loop over the diameter^2 offsets with core.filters.neighbor,
    never over pixels. A 512 x 512 image should take well under 2 seconds.
    """
    if diameter <= 1 or sigma_color <= 0.0 or sigma_space <= 0.0:
        return img.copy()

    src = io.to_float(img)
    radius = diameter // 2
    inv_two_space = -0.5 / (float(sigma_space) ** 2)
    inv_two_color = np.float32(-0.5 / (float(sigma_color) ** 2))

    acc = np.zeros_like(src, dtype=np.float32)
    w_sum = np.zeros(src.shape[:2], dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_w = np.float32(np.exp((dy * dy + dx * dx) * inv_two_space))
            shifted = filters.neighbor(src, dy, dx)
            diff = shifted - src
            color_dist_sq = np.sum(diff * diff, axis=-1)
            w = spatial_w * np.exp(color_dist_sq * inv_two_color)
            acc += w[..., None] * shifted
            w_sum += w

    out = acc / np.maximum(w_sum[..., None], np.float32(1e-8))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def enhance(img: np.ndarray, amount: float) -> np.ndarray:
    """Increase contrast and saturation. amount = 0 -> unchanged, 1 -> strong.
    (H, W, 3) uint8 -> (H, W, 3) uint8."""
    if amount <= 0.0:
        return img.copy()

    f = io.to_float(img)

    # Contrast enhancement around mean luminance
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    mean_lum = float(np.mean(lum))
    contrast = 1.0 + 0.5 * float(amount)
    f = mean_lum + contrast * (f - mean_lum)

    # Saturation enhancement in luminance-chroma space
    lum2 = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])[..., None]
    saturation = 1.0 + 0.6 * float(amount)
    f = lum2 + saturation * (f - lum2)

    return np.clip(np.round(f * 255.0), 0, 255).astype(np.uint8)