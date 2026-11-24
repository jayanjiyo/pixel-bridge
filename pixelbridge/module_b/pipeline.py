"""Module B pipeline: one entry point for every upscaling method.

OWNER: M1 (Leader) -- already implemented in the skeleton.
"""
from __future__ import annotations

import time

import numpy as np

from ..core import resize
from ..core.io import check_rgb
from . import scalex, slope, xbr

METHODS = ("nearest", "bilinear", "bicubic", "scale2x", "scale3x", "eagle", "xbr", "ours")

SUPPORTED_SCALES = {
    "nearest": (2, 3, 4),
    "bilinear": (2, 3, 4),
    "bicubic": (2, 3, 4),
    "scale2x": (2, 4),   # 4x = Scale2x applied twice (often called Scale4x)
    "scale3x": (3,),
    "eagle": (2, 4),     # 4x = Eagle applied twice
    "xbr": (2, 3, 4),
    "ours": (2, 3, 4),
}

MODES = ("palette", "smooth")  # only used by "xbr" and "ours"


def upscale(img: np.ndarray, scale: int = 4, method: str = "ours", mode: str = "palette") -> np.ndarray:
    """Enlarge pixel art. (H, W, 3) uint8 -> (scale*H, scale*W, 3) uint8."""
    check_rgb(img, "pixel art")
    if method not in SUPPORTED_SCALES:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if scale not in SUPPORTED_SCALES[method]:
        raise ValueError(f"{method} supports scales {SUPPORTED_SCALES[method]}, not {scale}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    h, w = img.shape[:2]

    if method == "nearest":
        return resize.resize_nearest(img, h * scale, w * scale)
    if method == "bilinear":
        return resize.resize_bilinear(img, h * scale, w * scale)
    if method == "bicubic":
        return resize.resize_bicubic(img, h * scale, w * scale)
    if method == "scale2x":
        out = scalex.scale2x(img)
        return scalex.scale2x(out) if scale == 4 else out
    if method == "scale3x":
        return scalex.scale3x(img)
    if method == "eagle":
        out = scalex.eagle(img)
        return scalex.eagle(out) if scale == 4 else out
    if method == "xbr":
        return xbr.xbr(img, scale=scale, mode=mode)
    return slope.upscale_slope(img, scale=scale, mode=mode)


def upscale_all(img: np.ndarray, scale: int = 4, methods=METHODS, mode: str = "palette") -> dict:
    """Run several methods for side-by-side comparison (GUI, report figures).

    Returns {method: {"image": array or None, "seconds": float, "error": str or None}}.
    Methods that do not support `scale` are skipped; methods that are not
    implemented yet are reported with an error message instead of crashing.
    """
    results = {}
    for m in methods:
        if scale not in SUPPORTED_SCALES[m]:
            continue
        t0 = time.perf_counter()
        try:
            out, err = upscale(img, scale, m, mode), None
        except NotImplementedError as e:
            out, err = None, str(e)
        results[m] = {"image": out, "seconds": time.perf_counter() - t0, "error": err}
    return results
