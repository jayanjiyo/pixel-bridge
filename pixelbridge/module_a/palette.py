"""Module A step 3: palette generation and color reduction.

OWNER: M2.   Tests: tests/test_module_a_front.py
Plan report section 4.4.

A palette is a (K, 3) uint8 array of RGB colors.
An index image is a (H, W) int32 array of palette indices; the final
pixel art is simply palette[indices].
"""
from __future__ import annotations

import numpy as np

from pixelbridge.core.color import lab_to_rgb, rgb_to_lab


def _hex(colors: list[str]) -> np.ndarray:
    return np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colors], dtype=np.uint8)


# Fixed palettes (plan report 4.4). Add more in Stage 3.
FIXED_PALETTES: dict[str, np.ndarray] = {
    # Four-color green palette in the style of early handheld game consoles
    "handheld_green": _hex(["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"]),
    "gray4": _hex(["#000000", "#555555", "#AAAAAA", "#FFFFFF"]),
    # 16-color fantasy console palette
    "pico8": _hex(["#000000", "#1D2B53", "#7E2553", "#008751", "#AB5236", "#5F574F",
                   "#C2C3C7", "#FFF1E8", "#FF004D", "#FFA300", "#FFEC27", "#00E436",
                   "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"]),
}


def median_cut(img: np.ndarray, n_colors: int) -> np.ndarray:
    """STAGE 1. Median cut (Heckbert 1982).

    Start with one box containing all pixels; repeatedly split the box with
    the largest channel range at the median of that channel until there are
    n_colors boxes (or no box can be split). Palette color = mean of the box.

    img: (H, W, 3) uint8 -> palette (K, 3) uint8 with K <= n_colors
    """
    n_colors = max(1, int(n_colors))
    pixels = img.reshape(-1, 3)
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    boxes: list[np.ndarray] = [pixels]
    while len(boxes) < n_colors:
        best_idx = -1
        best_range = 0
        best_ch = 0
        for idx, box in enumerate(boxes):
            if len(box) < 2:
                continue
            ranges = box.max(axis=0).astype(np.int32) - box.min(axis=0).astype(np.int32)
            ch = int(np.argmax(ranges))
            r = int(ranges[ch])
            if r > best_range:
                best_range = r
                best_idx = idx
                best_ch = ch

        if best_idx == -1 or best_range == 0:
            break

        box = boxes.pop(best_idx)
        order = np.argsort(box[:, best_ch], kind="mergesort")
        sorted_box = box[order]
        mid = len(sorted_box) // 2
        boxes.append(sorted_box[:mid])
        boxes.append(sorted_box[mid:])

    return np.array([np.round(b.mean(axis=0)) for b in boxes], dtype=np.uint8)


def kmeans_lab(img: np.ndarray, n_colors: int, init: np.ndarray | None = None,
               iterations: int = 10) -> np.ndarray:
    """STAGE 2. Refine a palette with k-means in CIELAB (core.color.rgb_to_lab).

    init : starting palette (K, 3) uint8, normally the median cut result.
           If None, use median_cut(img, n_colors).
    Keep a cluster that becomes empty at its previous center.
    returns (K, 3) uint8 RGB palette (convert the centers back with lab_to_rgb).
    """
    init_rgb = median_cut(img, n_colors) if init is None else np.asarray(init, dtype=np.uint8)
    pixels_lab = rgb_to_lab(img.reshape(-1, 3)).astype(np.float32)
    centers = rgb_to_lab(init_rgb).astype(np.float32)
    k = len(centers)

    prev_labels: np.ndarray | None = None
    for _ in range(iterations):
        diff = pixels_lab[:, None, :] - centers[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        labels = np.argmin(dist_sq, axis=1)

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels

        counts = np.bincount(labels, minlength=k)
        non_empty = counts > 0
        for ch in range(3):
            sums = np.bincount(labels, weights=pixels_lab[:, ch], minlength=k)
            centers[non_empty, ch] = sums[non_empty] / counts[non_empty]

    rgb_float = lab_to_rgb(centers)
    return np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)


def map_to_palette(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest palette color for every pixel (distance in CIELAB).

    img (H, W, 3) uint8, palette (K, 3) uint8 -> indices (H, W) int32
    Vectorize: distances of shape (H*W, K) are fine for pixel art sizes.
    """
    h, w = img.shape[:2]
    pixels_lab = rgb_to_lab(img.reshape(-1, 3)).astype(np.float32)
    pal_lab = rgb_to_lab(palette).astype(np.float32)

    diff = pixels_lab[:, None, :] - pal_lab[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    indices = np.argmin(dist_sq, axis=1).astype(np.int32)
    return indices.reshape(h, w)