"""Color similarity in YUV, shared by Scale2x/Eagle/xBR/our method.

OWNER: M4.   Tests: tests/test_module_b.py
Plan report sections 4.1 (step 1) and 4.7.
"""
from __future__ import annotations

import numpy as np

from ..core import color
from ..core.io import to_float

# Weights on |dY|, |dU|, |dV| (luminance weighted most), on 0..255-scaled YUV.
# Tune them and report the final values.
YUV_WEIGHTS = (48.0, 7.0, 6.0)


def yuv_distance(a: np.ndarray, b: np.ndarray, weights=YUV_WEIGHTS) -> np.ndarray:
    """Weighted color distance between two RGB arrays of the same shape (..., 3).

    Convert with core.color.rgb_to_yuv, then
    distance = wY*|dY| + wU*|dU| + wV*|dV|.
    Returns float32 of shape (...). Must be 0 for identical colors and symmetric.
    """
    # YUV on a 0..255 scale (Y in [0, 255], U/V centered at 0), as in the weights' comment
    ya = color.rgb_to_yuv(to_float(a)) * 255.0
    yb = color.rgb_to_yuv(to_float(b)) * 255.0
    diff = np.abs(ya - yb)
    w = np.asarray(weights, dtype=np.float32)
    return (diff * w).sum(axis=-1).astype(np.float32)


def similar(a: np.ndarray, b: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Boolean map: True where the colors are considered equal
    (yuv_distance <= threshold). Exact equality is threshold = 0."""
    return yuv_distance(a, b) <= threshold
