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


def _apply_separable(img: np.ndarray, wy: np.ndarray, wx: np.ndarray) -> np.ndarray:
    """out = Wy @ img @ Wx^T for every channel, then back to the input dtype.

    wy : (out_h, H) row weights, wx : (out_w, W) column weights. Every
    resize method here is separable, so it only has to build these two
    small weight matrices; the products are plain NumPy arithmetic.
    """
    src = img.astype(np.float64)
    out = np.tensordot(wy, src, axes=(1, 0))                 # (out_h, W, ...)
    out = np.moveaxis(np.tensordot(wx, out, axes=(1, 1)), 0, 1)  # (out_h, out_w, ...)
    if img.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(img.dtype)


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix: output pixel i covers [i*s, (i+1)*s) of the
    input with s = n_in / n_out; weight = overlap length / s."""
    s = n_in / n_out
    lo = np.arange(n_out)[:, None] * s                       # start of each output cell
    hi = lo + s
    j = np.arange(n_in)[None, :]                             # input pixel j covers [j, j+1)
    overlap = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0, None)
    return overlap / overlap.sum(axis=1, keepdims=True)


def resize_area(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Shrink by area averaging: each output pixel is the average of the
    input pixels it covers (weighted by overlap for non-integer factors).
    For an integer factor k this equals the mean of each k x k block.
    Only needs to support shrinking (out_h <= H and out_w <= W).
    """
    h, w = img.shape[:2]
    if out_h > h or out_w > w:
        raise ValueError(f"resize_area only shrinks: {h}x{w} -> {out_h}x{out_w}")
    return _apply_separable(img, _area_weights(h, out_h), _area_weights(w, out_w))


def resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest neighbor. Source index = floor(dst * in_size / out_size)
    (OpenCV INTER_NEAREST rule). For an integer upscale factor this is
    exactly np.repeat along both axes."""
    h, w = img.shape[:2]
    ys = np.minimum((np.arange(out_h) * h) // out_h, h - 1)
    xs = np.minimum((np.arange(out_w) * w) // out_w, w - 1)
    return img[ys][:, xs].copy()


def _interp_weights(n_in: int, n_out: int, taps: tuple[int, ...], kernel) -> np.ndarray:
    """(n_out, n_in) weight matrix for kernel interpolation along one axis.

    Output pixel i samples the input at src = (i + 0.5) * n_in / n_out - 0.5
    (pixel-center alignment, as in OpenCV). With x0 = floor(src) and
    t = src - x0, input pixel x0 + k gets weight kernel(k - t) for each k in
    `taps`. Indices outside the image are clamped to the edge (replicate).
    """
    src = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
    x0 = np.floor(src).astype(int)
    t = src - x0
    w = np.zeros((n_out, n_in))
    rows = np.arange(n_out)
    for k in taps:
        idx = np.clip(x0 + k, 0, n_in - 1)
        np.add.at(w, (rows, idx), kernel(k - t))
    return w


def _linear_kernel(d: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.abs(d), 0.0, None)


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear interpolation with pixel-center alignment and replicate
    border. Matches cv2.resize(..., INTER_LINEAR) within 1 for uint8
    upscaling (the test uses OpenCV only as a reference)."""
    h, w = img.shape[:2]
    return _apply_separable(img, _interp_weights(h, out_h, (0, 1), _linear_kernel),
                            _interp_weights(w, out_w, (0, 1), _linear_kernel))


def resize_bicubic(img: np.ndarray, out_h: int, out_w: int, a: float = -0.5) -> np.ndarray:
    """Bicubic (cubic convolution, Keys 1981) with parameter `a`.

    Keys uses a = -0.5 (our default, as in the plan report).
    OpenCV uses a = -0.75; with a = -0.75 the result must match
    cv2.resize(..., INTER_CUBIC) within 2 for uint8 upscaling.
    4x4 neighborhood, pixel-center alignment, replicate border.
    """
    raise NotImplementedError("TODO(M1): resize_bicubic")
