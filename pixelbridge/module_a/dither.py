"""Module A step 5: dithering (optional).

OWNER: M3.   Tests: tests/test_module_a_back.py
Plan report section 4.5.

Dithering replaces the plain nearest-color mapping (palette.map_to_palette)
when the user turns it on. Both dithering functions return palette INDICES.
"""
from __future__ import annotations

import numpy as np


def gradient_mask(img: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """Where dithering is allowed: True in smooth (gradient) regions,
    False near edges, so edges do not get noisy.

    Use the normalized Sobel gradient magnitude of the gray image
    (core.color.rgb_to_gray, core.filters.gradient_magnitude / SOBEL_MAX);
    True where it is below `threshold`.
    (H, W, 3) uint8 -> (H, W) bool
    """
    raise NotImplementedError("TODO(M3): gradient_mask")


def floyd_steinberg(img: np.ndarray, palette: np.ndarray, strength: float = 1.0,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """STAGE 1. Floyd-Steinberg error diffusion (1976).

    Scan rows top to bottom (serpentine order is a plus); map each pixel to
    the nearest palette color and push the error to the right / lower
    neighbors with weights 7/16, 3/16, 5/16, 1/16, multiplied by `strength`.
    Where mask is False, do not diffuse error (plain nearest color).
    A Python loop over pixels is acceptable here: pixel art is small (<= 128 wide).

    img (H, W, 3) uint8, palette (K, 3) uint8 -> indices (H, W) int32
    """
    raise NotImplementedError("TODO(M3): floyd_steinberg")


def bayer(img: np.ndarray, palette: np.ndarray, size: int = 4, strength: float = 1.0,
          mask: np.ndarray | None = None) -> np.ndarray:
    """STAGE 1. Ordered dithering with a Bayer threshold matrix (size 2, 4 or 8).

    Add (threshold - 0.5) * spread * strength to each pixel before choosing the
    nearest palette color, where spread is about the typical distance between
    palette colors. Where mask is False, add nothing. Fully vectorized.

    img (H, W, 3) uint8, palette (K, 3) uint8 -> indices (H, W) int32
    """
    raise NotImplementedError("TODO(M3): bayer")
