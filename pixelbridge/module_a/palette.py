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
    if len(pixels_lab) == 0:
        return init_rgb.copy()

    centers = rgb_to_lab(init_rgb).astype(np.float32)
    k = len(centers)

    def _run_lloyd(c: np.ndarray, max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prev_labels: np.ndarray | None = None
        labels = np.zeros(len(pixels_lab), dtype=np.intp)
        min_dist_sq = np.zeros(len(pixels_lab), dtype=np.float32)
        for _ in range(max_iter):
            diff = pixels_lab[:, None, :] - c[None, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            labels = np.argmin(dist_sq, axis=1)
            min_dist_sq = dist_sq[np.arange(len(pixels_lab)), labels]

            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels

            counts = np.bincount(labels, minlength=k)
            non_empty = counts > 0
            for ch in range(3):
                sums = np.bincount(labels, weights=pixels_lab[:, ch], minlength=k)
                c[non_empty, ch] = sums[non_empty] / counts[non_empty]
        return c, labels, min_dist_sq

    centers, labels, min_dist_sq = _run_lloyd(centers, iterations)

    # Merge near-duplicate active centers (Delta E < 8.0 in CIELAB) and reseed
    # the smaller duplicate to the pixel with largest quantization error.
    min_delta_e_sq = np.float32(8.0 ** 2)
    reseeded = False
    for _ in range(k):
        counts = np.bincount(labels, minlength=k)
        c_diff = centers[:, None, :] - centers[None, :, :]
        c_dist_sq = np.sum(c_diff * c_diff, axis=-1)
        np.fill_diagonal(c_dist_sq, np.inf)

        active = counts > 0
        valid_pairs = active[:, None] & active[None, :]
        c_dist_sq = np.where(valid_pairs, c_dist_sq, np.inf)

        min_idx = int(np.argmin(c_dist_sq))
        i, j = divmod(min_idx, k)
        if c_dist_sq[i, j] >= min_delta_e_sq:
            break

        victim = j if counts[i] >= counts[j] else i
        farthest_px = int(np.argmax(min_dist_sq))
        if min_dist_sq[farthest_px] < min_delta_e_sq:
            break

        centers[victim] = pixels_lab[farthest_px]
        reseeded = True
        diff = pixels_lab[:, None, :] - centers[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        labels = np.argmin(dist_sq, axis=1)
        min_dist_sq = dist_sq[np.arange(len(pixels_lab)), labels]

    if reseeded:
        centers, _, _ = _run_lloyd(centers, max(5, iterations // 2))

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