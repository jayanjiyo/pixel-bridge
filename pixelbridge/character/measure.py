"""Measure the few attributes that make a character look like the person.

Caricature research (Brennan 1985; Mo et al. 2004) shows that identity is
carried by a few distinctive features, not by the exact outline. So we do not
copy the face; we measure:

    expression  smile (mouth corners up) and smiling eyes (flat eye lines)
    hair        bangs type, part position, length, color
    colors      lit skin tone, shirt color
    light       which side of the face is darker

Facial lines come from XDoG (Winnemoeller et al. 2012): a sharpened
difference of Gaussians that keeps eyes, brows and the mouth as clean lines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import resize
from ..core.color import rgb_to_gray, rgb_to_lab
from ..core.filters import filter2d, gaussian_kernel, neighbor
from ..module_a.preprocess import bilateral_filter
from .detect import largest_component

WORK_W, WORK_H = 352, 368          # working size of the head crop (16 x the sprite head grid)


@dataclass
class Attributes:
    box: tuple[int, int, int, int]
    smile: bool
    happy_eyes: bool
    bangs: str                      # "none" | "curtain" | "full" | "side"
    part: float                     # hair part position across the head, 0..1 (0.5 = center)
    length: str                     # "short" | "medium" | "long"
    skin: np.ndarray                # RGB float
    hair: np.ndarray
    shirt: np.ndarray
    shadow_side: str | None         # "left" | "right" | None
    scores: dict                    # raw measurements, for debugging and the report


def xdog(gray: np.ndarray, sigma: float = 1.5, k: float = 1.6, p: float = 18.0,
         eps: float = 0.02, phi: float = 8.0) -> np.ndarray:
    """XDoG line image: 1 = paper, values toward 0 = lines."""
    def blur(x, s):
        size = int(2 * round(3 * s) + 1)
        return filter2d(x, gaussian_kernel(size, s))
    d = (1 + p) * blur(gray, sigma) - p * blur(gray, sigma * k)
    return np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))


def _components(mask: np.ndarray, max_iter: int = 300) -> list[dict]:
    """8-connected components with their bounding boxes and pixel lists."""
    h, w = mask.shape
    big = h * w + 1
    labels = np.where(mask, np.arange(h * w).reshape(h, w), big)
    for _ in range(max_iter):
        m = labels
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                m = np.minimum(m, neighbor(labels, dy, dx))
        m = np.where(mask, m, big)
        if np.array_equal(m, labels):
            break
        labels = m
    out = []
    for i in np.unique(labels[mask]):
        ys, xs = np.nonzero(labels == i)
        out.append(dict(n=len(ys), y0=ys.min(), y1=ys.max(), x0=xs.min(), x1=xs.max(),
                        cy=ys.mean(), cx=xs.mean(), ys=ys, xs=xs))
    return out


def _resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area averaging when shrinking, bilinear when the crop is smaller than the target."""
    h, w = img.shape[:2]
    if out_h <= h and out_w <= w:
        return resize.resize_area(img, out_h, out_w)
    return resize.resize_bilinear(img, out_h, out_w)


def _median(px: np.ndarray, fallback) -> np.ndarray:
    return np.median(px, axis=0) if len(px) else np.array(fallback, float)


def measure_attributes(photo: np.ndarray, box: tuple[int, int, int, int]) -> Attributes:
    x0, y0, x1, y1 = box
    crop = photo[y0:y1, x0:x1]
    work = bilateral_filter(_resize(crop, WORK_H, WORK_W), 7, 0.1, 3.0)
    lab = rgb_to_lab(work)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    rows = np.arange(WORK_H)[:, None]
    cols = np.arange(WORK_W)[None, :]

    # face extent from the largest skin region
    skin = largest_component((L > 35) & (a > 4) & (a < 35) & (b > 4) & (b < 40))
    if skin.sum() < 50:
        skin = np.zeros_like(skin)
        skin[int(WORK_H * 0.3):int(WORK_H * 0.85), int(WORK_W * 0.3):int(WORK_W * 0.7)] = True
    ys, xs = np.nonzero(skin)
    fx0, fx1 = np.percentile(xs, [5, 95])
    fy0, fy1 = np.percentile(ys, [2, 98])
    fcx, fw, fh = (fx0 + fx1) / 2, fx1 - fx0, fy1 - fy0
    hair = L < 35

    # facial lines (XDoG) inside the face -> eyes and mouth
    lines = xdog(rgb_to_gray(_resize(crop, WORK_H, WORK_W))) < 0.5
    inner = np.zeros_like(lines)
    inner[int(fy0 + 0.15 * fh):int(fy0 + 0.95 * fh), int(fx0 + 0.08 * fw):int(fx1 - 0.08 * fw)] = True
    comps = [c for c in _components(lines & inner) if c["n"] > 15]

    def eye(side):
        cand = [c for c in comps if fy0 + 0.20 * fh < c["cy"] < fy0 + 0.55 * fh
                and ((c["cx"] < fcx) if side < 0 else (c["cx"] > fcx))
                and (c["x1"] - c["x0"]) > 1.2 * (c["y1"] - c["y0"])]
        return max(cand, key=lambda c: c["n"]) if cand else None
    eyes = [e for e in (eye(-1), eye(1)) if e is not None]
    eye_y = np.mean([e["cy"] for e in eyes]) if eyes else fy0 + 0.40 * fh
    eye_open = (np.mean([(e["y1"] - e["y0"]) / max(1, e["x1"] - e["x0"]) for e in eyes])
                if eyes else 0.5)

    mouths = [c for c in comps if fy0 + 0.62 * fh < c["cy"] < fy0 + 0.92 * fh
              and abs(c["cx"] - fcx) < 0.25 * fw and (c["x1"] - c["x0"]) > 2 * (c["y1"] - c["y0"])]
    smile_score = 0.0
    if mouths:
        m = max(mouths, key=lambda c: c["x1"] - c["x0"])
        third = (m["x1"] - m["x0"]) / 3
        ends = np.mean([m["ys"][m["xs"] < m["x0"] + third].mean(), m["ys"][m["xs"] > m["x1"] - third].mean()])
        mid = m["ys"][(m["xs"] >= m["x0"] + third) & (m["xs"] <= m["x1"] - third)].mean()
        smile_score = (mid - ends) / max(1.0, m["x1"] - m["x0"])          # > 0: corners up

    # lit skin tone: bright 40% of the cheeks (avoids shadows)
    cheek = skin & (rows > eye_y + 0.05 * fh) & (rows < eye_y + 0.35 * fh)
    lum = L[cheek]
    skin_c = (_median(work[cheek][lum >= np.percentile(lum, 60)], (230, 190, 160))
              if cheek.sum() > 20 else _median(work[skin], (230, 190, 160)))
    hair_c = _median(work[hair & ~skin], (50, 35, 30))

    # hair part: brightest column inside the hair near the top center
    top = L[int(WORK_H * 0.08):int(WORK_H * 0.30), int(WORK_W * 0.35):int(WORK_W * 0.65)]
    hair_cols = (top < 45).mean(axis=0) > 0.3
    part = 0.5
    if hair_cols.any():
        prof = np.where(hair_cols, top.mean(axis=0), np.nan)
        c = int(np.nanargmax(prof))
        if np.nanmax(prof) - np.nanmedian(prof) > 6:
            part = (WORK_W * 0.35 + c) / WORK_W

    # bangs: hair coverage of the forehead band just above the eyes
    band = (rows > eye_y - 0.22 * fh) & (rows < eye_y - 0.07 * fh)
    center = band & (np.abs(cols - fcx) < 0.12 * fw)
    sides = band & (np.abs(cols - fcx) >= 0.12 * fw) & (np.abs(cols - fcx) < 0.42 * fw)
    cov_c, cov_s = float(hair[center].mean()), float(hair[sides].mean())
    cov_lr = (0.0, 0.0)
    if cov_c > 0.5:
        bangs = "full"
    elif cov_s > 0.30:
        left = float(hair[sides & (cols < fcx)].mean())
        right = float(hair[sides & (cols >= fcx)].mean())
        bangs = "side" if abs(left - right) > 0.6 else "curtain"
        cov_lr = (left, right)
        if bangs == "side":          # swept toward the covered side, parted on the other side
            part = 0.38 if left > right else 0.62
    else:
        bangs = "none"

    # hair length: how far hair continues below the chin beside the face, in the photo
    H, W = photo.shape[:2]
    sy = lambda v: int(y0 + v / WORK_H * (y1 - y0))
    sx = lambda u: int(x0 + u / WORK_W * (x1 - x0))
    chin = sy(fy1)
    face_h_px = (y1 - y0) * fh / WORK_H
    below = photo[chin:min(H, chin + int(1.2 * face_h_px))]
    if len(below) > 4:
        bl = rgb_to_lab(_resize(below, 24, max(1, below.shape[1] // max(1, below.shape[0] // 24))))
        side_cols = np.zeros(bl.shape[1], bool)
        cx_px = sx(fcx) * bl.shape[1] / W
        half = (x1 - x0) * fw / WORK_W / 2 * bl.shape[1] / W
        side_cols[max(0, int(cx_px - 2.2 * half)):max(0, int(cx_px - 0.8 * half))] = True
        side_cols[int(cx_px + 0.8 * half):int(cx_px + 2.2 * half)] = True
        dark_rows = (bl[:, side_cols, 0] < 35).mean(axis=1) > 0.15 if side_cols.any() else np.zeros(24, bool)
        reach = (np.nonzero(dark_rows)[0].max() + 1) / 24 if dark_rows.any() else 0.0
    else:
        reach = 0.0
    length = "short" if reach < 0.25 else "medium" if reach < 0.6 else "long"

    # shirt: below the chin, center columns, not skin and not hair
    r0 = min(H, chin + int(0.25 * (y1 - y0)))
    r1 = min(H, r0 + int(0.6 * (y1 - y0)))
    c0, c1 = max(0, sx(fcx - 0.45 * fw)), min(W, sx(fcx + 0.45 * fw))
    reg = photo[r0:r1, c0:c1].reshape(-1, 3)
    shirt = np.array([200, 60, 70], float)
    if len(reg) > 100:
        rl = rgb_to_lab(reg)
        chroma = np.hypot(rl[:, 1], rl[:, 2])
        skin_like = (rl[:, 1] > 5) & (rl[:, 1] < 30) & (rl[:, 2] > 8) & (rl[:, 2] < 35) & (rl[:, 0] > 45)
        hair_like = (rl[:, 0] < 35) & (chroma < 10)
        keep = (rl[:, 0] > 18) & ~skin_like & ~hair_like
        shirt = np.median(reg[keep] if keep.sum() > 50 else reg, axis=0)

    left_L = L[skin & (cols < fcx)].mean()
    right_L = L[skin & (cols >= fcx)].mean()
    shadow = "right" if right_L < left_L - 2 else "left" if left_L < right_L - 2 else None

    return Attributes(box=box, smile=smile_score > 0.02, happy_eyes=eye_open < 0.28, bangs=bangs,
                      part=float(part), length=length, skin=skin_c, hair=hair_c, shirt=shirt,
                      shadow_side=shadow,
                      scores=dict(smile=float(smile_score), eye_open=float(eye_open), eyes_found=len(eyes),
                                  bangs_center=cov_c, bangs_sides=cov_s, bangs_left=float(cov_lr[0]), bangs_right=float(cov_lr[1]), hair_reach=float(reach)))
