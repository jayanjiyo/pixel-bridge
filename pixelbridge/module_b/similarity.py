"""Color similarity in YUV, shared by Scale2x/Eagle/xBR/our method.

OWNER: M4.   Tests: tests/test_module_b.py
Plan report sections 4.1 (step 1) and 4.7.
"""
from __future__ import annotations

import numpy as np

from ..core.color import rgb_to_yuv

# Weights on |dY|, |dU|, |dV| (luminance weighted most), on 0..255-scaled YUV.
# Tune them and report the final values.
YUV_WEIGHTS = (48.0, 7.0, 6.0)


def yuv_distance(a: np.ndarray, b: np.ndarray, weights=YUV_WEIGHTS) -> np.ndarray:
    """Weighted color distance between two RGB arrays of the same shape (..., 3).

    Convert with core.color.rgb_to_yuv, then
    distance = wY*|dY| + wU*|dU| + wV*|dV|.
    Returns float32 of shape (...). Must be 0 for identical colors and symmetric.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    # YUV is in [0, 1] (U, V centered at 0); scale to 0..255 to match the weights.
    d = np.abs(rgb_to_yuv(a) - rgb_to_yuv(b)) * 255.0
    wy, wu, wv = weights
    return (wy * d[..., 0] + wu * d[..., 1] + wv * d[..., 2]).astype(np.float32)


def similar(a: np.ndarray, b: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Boolean map: True where the colors are considered equal
    (yuv_distance <= threshold). Exact equality is threshold = 0."""
    return yuv_distance(a, b) <= threshold
