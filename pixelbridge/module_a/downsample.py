"""Module A step 2: downsampling to the pixel art size.

OWNER: M2.   Tests: tests/test_module_a_front.py
Plan report section 4.3.

Block layout used by both functions (handles non-integer factors):
    ys = np.linspace(0, H, out_h + 1).astype(int)
    xs = np.linspace(0, W, out_w + 1).astype(int)
    output pixel (i, j) <- input block img[ys[i]:ys[i+1], xs[j]:xs[j+1]]
"""
from __future__ import annotations

import numpy as np


def target_size(h: int, w: int, out_width: int) -> tuple[int, int]:
    """Pixel art size (out_h, out_w) that keeps the aspect ratio.
    Implemented in the skeleton."""
    out_w = int(out_width)
    out_h = max(1, round(h * out_w / w))
    return out_h, out_w


def downsample_block_mode(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """STAGE 1. One representative color per block instead of the average.

    Group the colors of each block (e.g. quantize each channel to 3-4 bits
    and use the quantized value as the group key), take the largest group,
    and return the mean color of that group.
    A block that contains a single color must return exactly that color.

    (H, W, 3) uint8 -> (out_h, out_w, 3) uint8
    """
    raise NotImplementedError("TODO(M2): downsample_block_mode")


def downsample_edge_aware(img: np.ndarray, out_h: int, out_w: int,
                          edge_threshold: float = 0.25) -> np.ndarray:
    """STAGE 2. Like downsample_block_mode, but in blocks that contain a
    strong edge (normalized Sobel magnitude of the gray image above
    edge_threshold, see core.filters.SOBEL_MAX) prefer a reasonably large
    group on the darker side, so thin dark details (eyes, outlines) survive.

    (H, W, 3) uint8 -> (out_h, out_w, 3) uint8
    """
    raise NotImplementedError("TODO(M2): downsample_edge_aware")
