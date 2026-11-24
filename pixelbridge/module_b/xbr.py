"""xBR scaler (levels 1 and 2).

OWNER: M4.   Tests: tests/test_module_b.py
Plan report section 4.6. STAGE 2.
"""
from __future__ import annotations

import numpy as np


def xbr(img: np.ndarray, scale: int = 2, mode: str = "palette", level2: bool = True) -> np.ndarray:
    """xBR: for each corner of a pixel, compare the YUV color differences
    (similarity.yuv_distance) along the two diagonal directions in the 5x5
    neighborhood; if an edge is detected, fill the corner of the output block
    along that edge. Level 2 also detects edges with a slope of about 1:2.

    scale : 2, 3 or 4
    mode  : "palette" -> output uses only colors of the input (crisp)
            "smooth"  -> blend colors along the edge (anti-aliased)
    (H, W, 3) uint8 -> (scale*H, scale*W, 3) uint8
    A uniform image must stay uniform.
    """
    raise NotImplementedError("TODO(M4): xbr")
