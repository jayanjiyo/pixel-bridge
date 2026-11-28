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


# A skin region looks like a face when it is big enough, fills its bounding box
# (a face is a solid oval, not a thin strip or a scattered texture) and is a bit
# taller than wide. The region often includes the neck, chest or light hair, so
# the limits are loose. Checked on 7 frontal portraits (all pass, the smallest
# face covers 3% of a half-body shot) and on data/photos (16 of the 18 photos
# without a face fail; a city skyline and a close-up of a dog pass).
FACE_MIN_AREA = 0.025      # fraction of the image
FACE_MIN_FILL = 0.35       # region pixels / bounding box pixels
FACE_ASPECT = (0.9, 2.1)   # bounding box height / width


def looks_like_face(face: np.ndarray) -> bool:
    """True if the skin region (boolean mask) has the size and shape of a face."""
    n = int(face.sum())
    if n == 0:
        return False
    ys, xs = np.nonzero(face)
    bh, bw = np.ptp(ys) + 1, np.ptp(xs) + 1
    return bool(n / face.size >= FACE_MIN_AREA and n / (bh * bw) >= FACE_MIN_FILL
                and FACE_ASPECT[0] <= bh / bw <= FACE_ASPECT[1])


def find_head_box(photo: np.ndarray, return_found: bool = False):
    """Head box (x0, y0, x1, y1) from the largest skin region, or a centered box if none.

    With return_found=True, returns (box, found): found is False when no region
    looks like a face (see looks_like_face), so the caller can warn the user.
    """
    h, w = photo.shape[:2]
    s = 128 / max(h, w)
    small = resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    face = largest_component(skin_mask(small))
    found = looks_like_face(face)
    if face.sum() < 20:
        box = int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.7)
        return (box, False) if return_found else box
    ys, xs = np.nonzero(face)
    fy0, fy1 = ys.min() / s, ys.max() / s
    cx, fw = xs.mean() / s, (xs.max() - xs.min()) / s
    fh = fy1 - fy0
    x0, x1 = cx - 0.85 * fw, cx + 0.85 * fw
    y0, y1 = fy0 - 0.45 * fh, fy1 + 0.10 * fh
    box = (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))
    return (box, found) if return_found else box
