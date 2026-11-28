"""Find the head and facial features with simple, explainable rules.

- Skin mask: CIELAB thresholds (skin is light, slightly red and yellow).
- Head box: the largest skin region (our own connected-component labeling),
  expanded upward and sideways to include the hair.
- Facial features and other attributes: see measure.py.
"""
from __future__ import annotations

import numpy as np

from ..core import resize
from ..core.color import rgb_to_lab
from ..core.filters import neighbor


def skin_mask(img: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(img)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return (L > 40) & (L < 92) & (a > 5) & (a < 32) & (b > 6) & (b < 40)


def largest_component(mask: np.ndarray, max_iter: int = 400) -> np.ndarray:
    """Largest 4-connected region of a boolean mask (label propagation:
    every pixel repeatedly takes the smallest label among its neighbors)."""
    h, w = mask.shape
    big = h * w + 1
    labels = np.where(mask, np.arange(h * w).reshape(h, w), big)
    for _ in range(max_iter):
        m = labels
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            m = np.minimum(m, neighbor(labels, dy, dx))
        m = np.where(mask, m, big)
        if np.array_equal(m, labels):
            break
        labels = m
    ids, counts = np.unique(labels[mask], return_counts=True)
    if len(ids) == 0:
        return mask
    return labels == ids[np.argmax(counts)]


def find_head_box(photo: np.ndarray) -> tuple[int, int, int, int]:
    """Head box (x0, y0, x1, y1) from the largest skin region, or a centered box if none."""
    h, w = photo.shape[:2]
    s = 128 / max(h, w)
    small = resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    face = largest_component(skin_mask(small))
    if face.sum() < 20:
        return int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.7)
    ys, xs = np.nonzero(face)
    fy0, fy1 = ys.min() / s, ys.max() / s
    cx, fw = xs.mean() / s, (xs.max() - xs.min()) / s
    fh = fy1 - fy0
    x0, x1 = cx - 0.85 * fw, cx + 0.85 * fw
    y0, y1 = fy0 - 0.45 * fh, fy1 + 0.10 * fh
    return (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))
