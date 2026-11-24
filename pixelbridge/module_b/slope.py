"""OUR METHOD: extended slope handling for gentle edges (1:3, 1:4, ...).

OWNER: M4.   Tests: tests/test_module_b.py
Plan report section 4.7. STAGE 2 -- this is the main contribution of the
project, so keep debug outputs (slope maps) for the report figures.
"""
from __future__ import annotations

import numpy as np


def estimate_slopes(img: np.ndarray, max_run: int = 8) -> np.ndarray:
    """Measure the run lengths of same-colored pixels along edges over a range
    wider than xBR's 5x5 window and estimate the local edge slope.

    Returns an (H, W) float32 map: 0 = no edge, otherwise the run length n of
    a 1:n edge at that pixel (sign or a second map for direction is up to you).
    Used by upscale_slope and for visualization in the report.
    """
    raise NotImplementedError("TODO(M4): estimate_slopes")


def upscale_slope(img: np.ndarray, scale: int = 2, mode: str = "palette",
                  max_run: int = 8) -> np.ndarray:
    """Enlarge pixel art; near edges, fill the output pixels according to a
    straight line with the estimated slope (estimate_slopes) instead of a
    staircase. Elsewhere behave like xBR or nearest.

    mode : "palette" (only input colors) or "smooth" (blend along the edge)
    (H, W, 3) uint8 -> (scale*H, scale*W, 3) uint8
    A uniform image must stay uniform.
    """
    raise NotImplementedError("TODO(M4): upscale_slope")
