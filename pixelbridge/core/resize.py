"""Resizing / interpolation.

OWNER: M1 (Leader).   Tests: tests/test_core.py

Used by:
  - Module A preprocessing (resize_area to the working size)
  - Module B baselines: nearest, bilinear, bicubic (the methods we compare against)
  - evaluation

Common contract for every function:
  img   : (H, W, 3) or (H, W); uint8 or float32
  out_h, out_w : output size in pixels (numpy order: height first!)
  return: same number of channels and SAME dtype as the input
          (uint8 results are rounded and clipped to 0..255)
  Coordinates: pixel-center alignment like OpenCV,
          src = (dst + 0.5) * (in_size / out_size) - 0.5
          (except resize_nearest, see below)
"""
from __future__ import annotations

import numpy as np


def resize_area(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Shrink by area averaging: each output pixel is the average of the
    input pixels it covers (weighted by overlap for non-integer factors).
    For an integer factor k this equals the mean of each k x k block.
    Only needs to support shrinking (out_h <= H and out_w <= W).
    """
    raise NotImplementedError("TODO(M1): resize_area")


def resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest neighbor. Source index = floor(dst * in_size / out_size)
    (OpenCV INTER_NEAREST rule). For an integer upscale factor this is
    exactly np.repeat along both axes."""
    raise NotImplementedError("TODO(M1): resize_nearest")


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear interpolation with pixel-center alignment and replicate
    border. Must match cv2.resize(..., INTER_LINEAR) within 1 for uint8
    upscaling (the test uses OpenCV only as a reference)."""
    raise NotImplementedError("TODO(M1): resize_bilinear")


def resize_bicubic(img: np.ndarray, out_h: int, out_w: int, a: float = -0.5) -> np.ndarray:
    """Bicubic (cubic convolution, Keys 1981) with parameter `a`.

    Keys uses a = -0.5 (our default, as in the plan report).
    OpenCV uses a = -0.75; with a = -0.75 the result must match
    cv2.resize(..., INTER_CUBIC) within 2 for uint8 upscaling.
    4x4 neighborhood, pixel-center alignment, replicate border.
    """
    raise NotImplementedError("TODO(M1): resize_bicubic")
