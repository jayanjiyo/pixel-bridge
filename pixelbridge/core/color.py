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

from .io import to_float


# BT.601 luminance weights
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# sRGB (D65) -> XYZ matrix and D65 reference white (same constants as OpenCV)
_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE = np.array([0.950456, 1.0, 1.088754], dtype=np.float64)
_EPS = 0.008856          # (6/29)^3: below this the Lab curve is linear
_KAPPA = 903.3           # slope of L in the linear part
_LIN_A, _LIN_B = 7.787, 16.0 / 116.0


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance with BT.601 weights: 0.299 R + 0.587 G + 0.114 B.

    Returns float32 with the last axis removed, values in [0, 1].
    """
    return (to_float(rgb) @ _GRAY).astype(np.float32)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """RGB -> YUV (BT.601).

        Y = 0.299 R + 0.587 G + 0.114 B      in [0, 1]
        U = 0.492 (B - Y)                    centered at 0
        V = 0.877 (R - Y)                    centered at 0

    Returns float32 of the same shape.
    """
    f = to_float(rgb)
    y = f @ _GRAY
    u = 0.492 * (f[..., 2] - y)
    v = 0.877 * (f[..., 0] - y)
    return np.stack([y, u, v], axis=-1).astype(np.float32)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_yuv. Returns float32 RGB clipped to [0, 1]."""
    yuv = yuv.astype(np.float32)
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    b = y + u / 0.492
    r = y + v / 0.877
    g = (y - _GRAY[0] * r - _GRAY[2] * b) / _GRAY[1]
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> CIELAB (D65 white point).

    Steps: undo the sRGB gamma -> linear RGB -> XYZ -> L*a*b*.
    Returns float32 with L in [0, 100] and a, b roughly in [-128, 127].
    Matches cv2.cvtColor(float_rgb, cv2.COLOR_RGB2Lab) within 0.5
    (the test uses OpenCV only as a reference).
    """
    linear = _srgb_to_linear(to_float(rgb).astype(np.float64))
    t = (linear @ _RGB2XYZ.T) / _WHITE                       # X/Xn, Y/Yn, Z/Zn
    f = np.where(t > _EPS, np.cbrt(t), _LIN_A * t + _LIN_B)
    ty = t[..., 1]
    L = np.where(ty > _EPS, 116.0 * f[..., 1] - 16.0, _KAPPA * ty)
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab. Returns float32 sRGB clipped to [0, 1]."""
    lab = lab.astype(np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    t = np.where(f > np.cbrt(_EPS), f ** 3, (f - _LIN_B) / _LIN_A)
    linear = (t * _WHITE) @ _XYZ2RGB.T
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0).astype(np.float32)
