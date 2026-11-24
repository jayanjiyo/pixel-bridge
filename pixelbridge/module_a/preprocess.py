"""Module A step 1: preprocessing.

OWNER: M2.   Tests: tests/test_module_a_front.py
Plan report section 4.2.
"""
from __future__ import annotations

import numpy as np

from ..core import resize


def preprocess(photo: np.ndarray, working_size: int = 512, smooth: bool = True,
               enhance_amount: float = 0.0) -> np.ndarray:
    """Resize so that the longer side is at most `working_size` (area
    averaging), then optionally smooth and enhance. Implemented in the
    skeleton; it only calls the functions below.
    """
    h, w = photo.shape[:2]
    longer = max(h, w)
    if longer > working_size:
        s = working_size / longer
        photo = resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    if smooth:
        photo = bilateral_filter(photo)
    if enhance_amount > 0:
        photo = enhance(photo, enhance_amount)
    return photo


def bilateral_filter(img: np.ndarray, diameter: int = 7, sigma_color: float = 0.1,
                     sigma_space: float = 3.0) -> np.ndarray:
    """Edge-preserving smoothing (Tomasi and Manduchi 1998).

    img         : (H, W, 3) uint8
    diameter    : window size (odd)
    sigma_color : range sigma on float colors in [0, 1] (color distance in RGB
                  or CIELAB, your choice -- document it in the report)
    sigma_space : spatial sigma in pixels
    returns     : (H, W, 3) uint8

    Speed: loop over the diameter^2 offsets with core.filters.neighbor,
    never over pixels. A 512 x 512 image should take well under 2 seconds.
    """
    raise NotImplementedError("TODO(M2): bilateral_filter")


def enhance(img: np.ndarray, amount: float) -> np.ndarray:
    """Increase contrast and saturation. amount = 0 -> unchanged, 1 -> strong.
    (H, W, 3) uint8 -> (H, W, 3) uint8."""
    raise NotImplementedError("TODO(M2): enhance")
