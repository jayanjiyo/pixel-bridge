"""Neighborhood operations: filtering, gradients, and the `neighbor` helper.

OWNER: M1 (Leader).   Tests: tests/test_core.py

Performance tip for EVERYONE: avoid Python loops over pixels. Loop over the
few kernel offsets instead and use `neighbor(img, dy, dx)` to get a whole
shifted copy of the image at once. Example (3x3 box blur):

    acc = sum(neighbor(img, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    blurred = acc / 9
"""
from __future__ import annotations

import numpy as np

# Largest possible Sobel magnitude for a gray image in [0, 1].
# "Normalized gradient magnitude" everywhere in the project means
# gradient_magnitude(gray) / SOBEL_MAX, which lies in [0, 1].
SOBEL_MAX = float(4 * np.sqrt(2))


def neighbor(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return an array where out[y, x] = img[y + dy, x + dx].

    Pixels outside the image are replaced by the nearest edge pixel
    (replicate border). Works for (H, W) and (H, W, C) arrays of any dtype.
    Implemented in the skeleton.
    """
    h, w = img.shape[:2]
    pad = max(abs(dy), abs(dx))
    if pad == 0:
        return img.copy()
    pad_width = ((pad, pad), (pad, pad)) + ((0, 0),) * (img.ndim - 2)
    padded = np.pad(img, pad_width, mode="edge")
    return padded[pad + dy: pad + dy + h, pad + dx: pad + dx + w]


def filter2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlate `img` with `kernel` (same as cv2.filter2D, NOT flipped).

    img    : (H, W) or (H, W, C); converted to float32 (uint8 stays 0..255 scale)
    kernel : (kh, kw) with odd kh, kw
    Border : replicate (use `neighbor`)
    Returns float32 of the same shape as img.
    """
    raise NotImplementedError("TODO(M1): filter2d")


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """2D Gaussian kernel (size x size, size odd), float32, sums to 1."""
    raise NotImplementedError("TODO(M1): gaussian_kernel")


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel derivatives of a (H, W) image.

    gx uses [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] (positive when the image
    gets brighter to the right), gy is its transpose (positive when the
    image gets brighter downward). Returns (gx, gy) as float32.
    """
    raise NotImplementedError("TODO(M1): sobel")


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """sqrt(gx^2 + gy^2) from `sobel`. Returns float32 (H, W)."""
    raise NotImplementedError("TODO(M1): gradient_magnitude")
