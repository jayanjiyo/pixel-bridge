"""Color space conversions.

OWNER: M1 (Leader).   Tests: tests/test_core.py

All functions accept either uint8 values in [0, 255] or float values in
[0, 1], and any array whose LAST axis has length 3. So the same function
works on an image (H, W, 3) and on a palette (K, 3).
Use `core.io.to_float` at the start of each function.

Users:
    rgb_to_gray : evaluation, edge detection in Module A
    rgb_to_yuv  : Module B color similarity (xBR compares colors in YUV)
    rgb_to_lab  : Module A palette (k-means in CIELAB), evaluation (Delta E)
"""
from __future__ import annotations

import numpy as np

from .io import to_float  # noqa: F401  (use it in the implementations)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance with BT.601 weights: 0.299 R + 0.587 G + 0.114 B.

    Returns float32 with the last axis removed, values in [0, 1].
    """
    raise NotImplementedError("TODO(M1): rgb_to_gray")


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """RGB -> YUV (BT.601).

        Y = 0.299 R + 0.587 G + 0.114 B      in [0, 1]
        U = 0.492 (B - Y)                    centered at 0
        V = 0.877 (R - Y)                    centered at 0

    Returns float32 of the same shape.
    """
    raise NotImplementedError("TODO(M1): rgb_to_yuv")


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_yuv. Returns float32 RGB clipped to [0, 1]."""
    raise NotImplementedError("TODO(M1): yuv_to_rgb")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> CIELAB (D65 white point).

    Steps: undo the sRGB gamma -> linear RGB -> XYZ -> L*a*b*.
    Returns float32 with L in [0, 100] and a, b roughly in [-128, 127].
    Must match cv2.cvtColor(float_rgb, cv2.COLOR_RGB2Lab) within 0.5
    (the test uses OpenCV only as a reference; do not call it here).
    """
    raise NotImplementedError("TODO(M1): rgb_to_lab")


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab. Returns float32 sRGB clipped to [0, 1]."""
    raise NotImplementedError("TODO(M1): lab_to_rgb")
