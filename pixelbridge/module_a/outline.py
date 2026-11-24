"""Module A step 4: one-pixel outlines.

OWNER: M3.   Tests: tests/test_module_a_back.py
Plan report section 4.5.
"""
from __future__ import annotations

import numpy as np


def darker_index(palette: np.ndarray) -> np.ndarray:
    """For each palette color, the index of a darker palette color to use
    for outlines (e.g. the darkest color with a similar hue, or simply the
    nearest color with lower CIELAB L*). The darkest color maps to itself.

    palette (K, 3) uint8 -> (K,) int32
    """
    raise NotImplementedError("TODO(M3): darker_index")


def add_outline(indices: np.ndarray, palette: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    """Draw a one-pixel dark outline where neighboring colors differ a lot.

    For each pair of 4-neighbors whose CIELAB distance is above `threshold`,
    replace ONE side (e.g. the brighter pixel, or the pixel of the object
    rather than the background -- decide and document) with
    darker_index(palette)[...]. Only palette colors may be used, so the
    result is again a valid index image. A uniform image must not change.

    indices (H, W) int32, palette (K, 3) uint8 -> indices (H, W) int32
    """
    raise NotImplementedError("TODO(M3): add_outline")
