"""Module A step 3: palette generation and color reduction.

OWNER: M2.   Tests: tests/test_module_a_front.py
Plan report section 4.4.

A palette is a (K, 3) uint8 array of RGB colors.
An index image is a (H, W) int32 array of palette indices; the final
pixel art is simply palette[indices].
"""
from __future__ import annotations

import numpy as np


def _hex(colors: list[str]) -> np.ndarray:
    return np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colors], dtype=np.uint8)


# Fixed palettes (plan report 4.4). Add more in Stage 3.
FIXED_PALETTES: dict[str, np.ndarray] = {
    # Four-color green palette in the style of early handheld game consoles
    "handheld_green": _hex(["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"]),
    "gray4": _hex(["#000000", "#555555", "#AAAAAA", "#FFFFFF"]),
    # 16-color fantasy console palette
    "pico8": _hex(["#000000", "#1D2B53", "#7E2553", "#008751", "#AB5236", "#5F574F",
                   "#C2C3C7", "#FFF1E8", "#FF004D", "#FFA300", "#FFEC27", "#00E436",
                   "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"]),
}


def median_cut(img: np.ndarray, n_colors: int) -> np.ndarray:
    """STAGE 1. Median cut (Heckbert 1982).

    Start with one box containing all pixels; repeatedly split the box with
    the largest channel range at the median of that channel until there are
    n_colors boxes (or no box can be split). Palette color = mean of the box.

    img: (H, W, 3) uint8 -> palette (K, 3) uint8 with K <= n_colors
    """
    raise NotImplementedError("TODO(M2): median_cut")


def kmeans_lab(img: np.ndarray, n_colors: int, init: np.ndarray | None = None,
               iterations: int = 10) -> np.ndarray:
    """STAGE 2. Refine a palette with k-means in CIELAB (core.color.rgb_to_lab).

    init : starting palette (K, 3) uint8, normally the median cut result.
           If None, use median_cut(img, n_colors).
    Keep a cluster that becomes empty at its previous center.
    returns (K, 3) uint8 RGB palette (convert the centers back with lab_to_rgb).
    """
    raise NotImplementedError("TODO(M2): kmeans_lab")


def map_to_palette(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest palette color for every pixel (distance in CIELAB).

    img (H, W, 3) uint8, palette (K, 3) uint8 -> indices (H, W) int32
    Vectorize: distances of shape (H*W, K) are fine for pixel art sizes.
    """
    raise NotImplementedError("TODO(M2): map_to_palette")
