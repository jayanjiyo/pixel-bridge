"""Module A step 6: cleanup rules used by pixel artists.

OWNER: M3.   Tests: tests/test_module_a_back.py
Plan report section 4.5.

Both functions work on index images (H, W) int32 and return a new index
image of the same shape; they never add new palette entries.
"""
from __future__ import annotations

import numpy as np


def remove_isolated(indices: np.ndarray) -> np.ndarray:
    """A pixel whose index differs from ALL eight neighbors is replaced by
    the most common neighbor index (mode filter). Image borders count the
    replicated edge as neighbors (core.filters.neighbor)."""
    raise NotImplementedError("TODO(M3): remove_isolated")


def fix_one_pixel_lines(indices: np.ndarray) -> np.ndarray:
    """Make diagonal lines exactly one pixel wide.

    In a diagonal line, a pixel that makes an L-shaped corner is redundant:
    it has a same-color neighbor horizontally AND vertically, the fourth
    pixel of that 2x2 square has another color, and the line does not go
    on straight through it. Replace it with the surrounding (background)
    color; the line stays 8-connected.

        X . . .        X . . .
        . X X .   ->   . X . .        (45-degree line with an extra pixel)
        . . X .        . . X .

    Must NOT change: filled areas, straight lines, and the right-angle
    corners of rectangles (there the line continues straight on both sides).
    """
    raise NotImplementedError("TODO(M3): fix_one_pixel_lines")
