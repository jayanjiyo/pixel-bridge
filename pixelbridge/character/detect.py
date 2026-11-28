"""Find the head and facial features with simple, explainable rules.

- Skin mask: CIELAB thresholds (skin is light, slightly red and yellow).
- Head box: the largest skin region (our own connected-component labeling),
  expanded upward and sideways to include the hair.
- Eyes / mouth (find_features): projection profiles -- darkest row and
  columns for the eyes, reddest row for the mouth. Used for analysis and the
  report; the sprite itself draws the features at template positions,
  because the measured positions are not reliable enough at 22 pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import resize
from ..core.color import rgb_to_gray, rgb_to_lab
from ..core.filters import neighbor


@dataclass
class Face:
    box: tuple[int, int, int, int]   # head box in the photo: x0, y0, x1, y1
    eye_y: float                     # positions relative to the head box, 0..1
    eye_dx: float                    # distance of each eye from the face center, 0..1 of box width
    center_x: float
    mouth_y: float


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


def find_features(photo: np.ndarray, box: tuple[int, int, int, int]) -> Face:
    x0, y0, x1, y1 = box
    crop = photo[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    skin = skin_mask(crop)
    cols = np.nonzero(skin.any(axis=0))[0]
    center = float(np.mean(np.nonzero(skin)[1])) if skin.any() else cw / 2
    fx0 = cols.min() if len(cols) else int(cw * 0.25)
    fx1 = cols.max() if len(cols) else int(cw * 0.75)

    gray = rgb_to_gray(crop)
    band0, band1 = int(0.30 * ch), int(0.62 * ch)
    inner = slice(int(fx0 + 0.15 * (fx1 - fx0)), int(fx1 - 0.15 * (fx1 - fx0)) + 1)
    eye_y = band0 + int(np.argmin(gray[band0:band1, inner].mean(axis=1)))
    strip = gray[max(0, eye_y - 6): eye_y + 7].mean(axis=0)
    c = int(center)
    lx = fx0 + int(np.argmin(strip[fx0:c])) if c > fx0 else int(cw * 0.35)
    rx = c + int(np.argmin(strip[c:fx1 + 1])) if fx1 > c else int(cw * 0.65)
    dx = min(abs(center - lx), abs(rx - center))
    if dx < 0.08 * cw or dx > 0.30 * cw:            # unreliable -> typical proportion
        dx = 0.17 * cw

    lab = rgb_to_lab(crop)
    m0, m1 = int(0.66 * ch), int(0.92 * ch)
    mouth_cols = slice(max(0, c - int(0.1 * cw)), c + int(0.1 * cw) + 1)
    mouth_y = m0 + int(np.argmax(lab[m0:m1, mouth_cols, 1].mean(axis=1)))
    return Face(box=box, eye_y=eye_y / ch, eye_dx=dx / cw, center_x=center / cw, mouth_y=mouth_y / ch)
