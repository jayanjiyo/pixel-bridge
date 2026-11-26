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

from pixelbridge.core.color import rgb_to_gray
from pixelbridge.core.filters import SOBEL_MAX, gradient_magnitude
from pixelbridge.core.io import to_float


def target_size(h: int, w: int, out_width: int) -> tuple[int, int]:
    """Pixel art size (out_h, out_w) that keeps the aspect ratio.
    Implemented in the skeleton."""
    out_w = int(out_width)
    out_h = max(1, round(h * out_w / w))
    return out_h, out_w


def _group_block_colors(block: np.ndarray, bits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Quantize RGB pixels in `block` to `bits` per channel and compute
    each color group's pixel count and mean RGB color.
    """
    pixels = block.reshape(-1, 3)
    shift = 8 - bits
    q = pixels.astype(np.int32) >> shift
    keys = (q[:, 0] << (2 * bits)) | (q[:, 1] << bits) | q[:, 2]

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    num_groups = len(counts)

    mean_r = np.bincount(inverse, weights=pixels[:, 0], minlength=num_groups) / counts
    mean_g = np.bincount(inverse, weights=pixels[:, 1], minlength=num_groups) / counts
    mean_b = np.bincount(inverse, weights=pixels[:, 2], minlength=num_groups) / counts
    group_means = np.stack([mean_r, mean_g, mean_b], axis=1)
    return counts, group_means


def downsample_block_mode(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """STAGE 1. One representative color per block instead of the average.

    Group the colors of each block (e.g. quantize each channel to 3-4 bits
    and use the quantized value as the group key), take the largest group,
    and return the mean color of that group.
    A block that contains a single color must return exactly that color.

    (H, W, 3) uint8 -> (out_h, out_w, 3) uint8
    """
    h, w = img.shape[:2]
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for i in range(out_h):
        y0 = ys[i]
        y1 = min(h, max(y0 + 1, ys[i + 1]))
        for j in range(out_w):
            x0 = xs[j]
            x1 = min(w, max(x0 + 1, xs[j + 1]))
            block = img[y0:y1, x0:x1]
            counts, group_means = _group_block_colors(block)
            best = int(np.argmax(counts))
            out[i, j] = np.clip(np.round(group_means[best]), 0, 255).astype(np.uint8)
    return out


def downsample_edge_aware(img: np.ndarray, out_h: int, out_w: int,
                          edge_threshold: float = 0.25) -> np.ndarray:
    """STAGE 2. Like downsample_block_mode, but in blocks that contain a
    strong edge (normalized Sobel magnitude of the gray image above
    edge_threshold, see core.filters.SOBEL_MAX) prefer a reasonably large
    group on the darker side, so thin dark details (eyes, outlines) survive.

    (H, W, 3) uint8 -> (out_h, out_w, 3) uint8
    """
    h, w = img.shape[:2]
    gray = rgb_to_gray(to_float(img))
    if gray.max() > 1.0:
        gray = gray / 255.0
    norm_grad = gradient_magnitude(gray) / SOBEL_MAX

    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for i in range(out_h):
        y0 = ys[i]
        y1 = min(h, max(y0 + 1, ys[i + 1]))
        for j in range(out_w):
            x0 = xs[j]
            x1 = min(w, max(x0 + 1, xs[j + 1]))
            block = img[y0:y1, x0:x1]
            counts, group_means = _group_block_colors(block)
            best = int(np.argmax(counts))

            block_grad = norm_grad[y0:y1, x0:x1]
            if block_grad.max() > edge_threshold and len(counts) > 1:
                luma = (
                    0.299 * group_means[:, 0]
                    + 0.587 * group_means[:, 1]
                    + 0.114 * group_means[:, 2]
                )
                n_pixels = block.shape[0] * block.shape[1]
                block_luma = float(np.sum(luma * counts) / n_pixels)
                min_count = max(2, int(np.ceil(0.08 * n_pixels)))

                dark_candidates = np.where((counts >= min_count) & (luma < block_luma - 10.0))[0]
                if len(dark_candidates) > 0:
                    best = int(dark_candidates[np.argmin(luma[dark_candidates])])

            out[i, j] = np.clip(np.round(group_means[best]), 0, 255).astype(np.uint8)
    return out