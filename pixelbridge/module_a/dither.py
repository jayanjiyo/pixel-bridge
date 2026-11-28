"""Module A step 5: dithering (optional).

OWNER: M3.   Tests: tests/test_module_a_back.py
Plan report section 4.5.

Dithering replaces the plain nearest-color mapping (palette.map_to_palette)
when the user turns it on. Both dithering functions return palette INDICES.

Design choices
--------------
* The nearest palette color is chosen in CIELAB, the same distance that
  palette.map_to_palette uses, so dithering with strength 0 (or a mask that
  is all False) gives the same result as no dithering.
* The quantization error is measured and spread in RGB (0..255), because
  dithering mixes colors on the screen, which is additive in RGB.
"""
from __future__ import annotations

import numpy as np

from ..core.color import rgb_to_gray, rgb_to_lab
from ..core.filters import SOBEL_MAX, gradient_magnitude


def gradient_mask(img: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """Where dithering is allowed: True in smooth (gradient) regions,
    False near edges, so edges do not get noisy.

    Use the normalized Sobel gradient magnitude of the gray image
    (core.color.rgb_to_gray, core.filters.gradient_magnitude / SOBEL_MAX);
    True where it is below `threshold`.
    (H, W, 3) uint8 -> (H, W) bool
    """
    magnitude = gradient_magnitude(rgb_to_gray(img)) / SOBEL_MAX
    return magnitude < threshold


# ---------------------------------------------------------------- helpers
def _nearest_lab(colors_rgb: np.ndarray, pal_lab: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (CIELAB distance) for each color.

    colors_rgb (..., 3) float in 0..255, pal_lab (K, 3) -> (...) int32
    """
    lab = rgb_to_lab(np.clip(colors_rgb, 0.0, 255.0) / 255.0)
    diff = lab[..., None, :] - pal_lab
    return np.argmin(np.sum(diff * diff, axis=-1), axis=-1).astype(np.int32)


def _palette_spread(palette: np.ndarray) -> float:
    """Typical per-channel step between palette colors (0..255 scale).

    Mean RGB distance from each palette color to its nearest other color,
    divided by sqrt(3) because the Bayer offset is added to all three
    channels at once (an offset d moves a color by d * sqrt(3)).
    For black/white this gives 255, the classic two-level case.
    """
    pal = palette.astype(np.float64)
    if len(pal) < 2:
        return 0.0
    d = np.sqrt(np.sum((pal[:, None, :] - pal[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean() / np.sqrt(3.0))


def bayer_matrix(size: int) -> np.ndarray:
    """Bayer threshold matrix (size 2, 4 or 8) with values in (0, 1).

    Built recursively: M(2n) = [[4M, 4M + 2], [4M + 3, 4M + 1]],
    thresholds = (M + 0.5) / n^2.
    """
    if size not in (2, 4, 8):
        raise ValueError(f"Bayer size must be 2, 4 or 8, got {size}")
    m = np.array([[0, 2], [3, 1]], dtype=np.float64)
    while m.shape[0] < size:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return ((m + 0.5) / (size * size)).astype(np.float32)


# ---------------------------------------------------------------- dithering
_FS_WEIGHTS = ((0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16))


def floyd_steinberg(img: np.ndarray, palette: np.ndarray, strength: float = 1.0,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """STAGE 1. Floyd-Steinberg error diffusion (1976).

    Scan rows top to bottom (serpentine order is a plus); map each pixel to
    the nearest palette color and push the error to the right / lower
    neighbors with weights 7/16, 3/16, 5/16, 1/16, multiplied by `strength`.
    Where mask is False, do not diffuse error (plain nearest color).
    A Python loop over pixels is acceptable here: pixel art is small (<= 128 wide).

    Serpentine order is used: even rows go left -> right, odd rows right -> left
    (the weights are mirrored), which avoids diagonal "worm" artifacts.
    A pixel where mask is False ignores the error it received, takes the plain
    nearest color of its original value, and passes no error on.

    img (H, W, 3) uint8, palette (K, 3) uint8 -> indices (H, W) int32
    """
    h, w = img.shape[:2]
    pal_rgb = palette.astype(np.float64)
    pal_lab = rgb_to_lab(palette).astype(np.float64)
    allowed = np.ones((h, w), bool) if mask is None else np.asarray(mask, bool)

    # Plain nearest colors (vectorized); used where diffusion is not allowed
    # and as the answer everywhere when strength is 0.
    plain = _nearest_lab(img.astype(np.float64), pal_lab)
    if strength == 0 or not allowed.any():
        return plain

    work = img.astype(np.float64)              # pixel values + received error
    out = plain.copy()
    for y in range(h):
        forward = y % 2 == 0
        xs = range(w) if forward else range(w - 1, -1, -1)
        sign = 1 if forward else -1
        for x in xs:
            if not allowed[y, x]:
                continue                        # keep plain nearest, no error
            old = work[y, x]
            k = int(_nearest_lab(old, pal_lab))
            out[y, x] = k
            err = (old - pal_rgb[k]) * strength
            for dy, dx, wgt in _FS_WEIGHTS:
                ny, nx = y + dy, x + sign * dx
                if 0 <= ny < h and 0 <= nx < w:
                    work[ny, nx] += err * wgt
    return out


def bayer(img: np.ndarray, palette: np.ndarray, size: int = 4, strength: float = 1.0,
          mask: np.ndarray | None = None) -> np.ndarray:
    """STAGE 1. Ordered dithering with a Bayer threshold matrix (size 2, 4 or 8).

    Add (threshold - 0.5) * spread * strength to each pixel before choosing the
    nearest palette color, where spread is about the typical distance between
    palette colors. Where mask is False, add nothing. Fully vectorized.

    spread: see _palette_spread (mean nearest-neighbor distance of the
    palette, per channel). The same offset is added to R, G and B.

    img (H, W, 3) uint8, palette (K, 3) uint8 -> indices (H, W) int32
    """
    h, w = img.shape[:2]
    thresholds = bayer_matrix(size)
    reps = (-(-h // size), -(-w // size))                   # ceil division
    t = np.tile(thresholds, reps)[:h, :w]
    offset = (t - 0.5) * _palette_spread(palette) * strength
    if mask is not None:
        offset = np.where(np.asarray(mask, bool), offset, 0.0)
    shifted = img.astype(np.float64) + offset[..., None]
    pal_lab = rgb_to_lab(palette).astype(np.float64)
    return _nearest_lab(shifted, pal_lab)
