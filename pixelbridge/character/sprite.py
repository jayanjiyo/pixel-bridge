"""Build a 2-head-tall (chibi) pixel character from a photo.

From the photo:  head box, face center, hair silhouette and color, skin color, shirt color.
From templates:  round face, big 2x3 eyes with a highlight, blush, small body.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import resize
from ..core.color import rgb_to_lab
from ..core.filters import neighbor
from .detect import Face, find_features, find_head_box, skin_mask

HEAD_W, HEAD_H = 22, 23
BODY_W, BODY_H = 16, 10
PANTS = (60, 70, 110)
SHOES = (40, 40, 45)


@dataclass
class Character:
    image: np.ndarray   # (h, w, 3) uint8
    mask: np.ndarray    # (h, w) bool, True where the character is
    face: Face


def _median(img, m, fallback):
    px = img[m]
    return np.median(px, axis=0) if len(px) else np.array(fallback, float)


def _shirt_color(photo, box):
    h, w = photo.shape[:2]
    x0, _, x1, y1 = box
    region = photo[min(h - 1, y1):, max(0, x0):x1].reshape(-1, 3)
    if len(region) < 50:
        return np.array([200, 60, 70], float)
    lab = rgb_to_lab(region)
    colorful = np.hypot(lab[:, 1], lab[:, 2]) > 25
    return np.median(region[colorful] if colorful.sum() > 50 else region, axis=0)


def make_character(photo: np.ndarray, head_box: tuple[int, int, int, int] | None = None,
                   symmetric_hair: bool = True) -> Character:
    """photo (H, W, 3) uint8 -> Character (about 24 x 34 pixels)."""
    box = head_box or find_head_box(photo)
    face = find_features(photo, box)
    x0, y0, x1, y1 = box
    crop = photo[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    lab = rgb_to_lab(crop)
    yy, xx = np.mgrid[:ch, :cw]
    head_ell = ((xx - 0.5 * cw) / (0.48 * cw)) ** 2 + ((yy - 0.5 * ch) / (0.53 * ch)) ** 2 < 1
    face_in = ((xx - face.center_x * cw) / (0.24 * cw)) ** 2 + ((yy - 0.60 * ch) / (0.34 * ch)) ** 2 < 1
    hair = (lab[..., 0] < 35) & head_ell & ~face_in
    skin = skin_mask(crop)

    skin_c = _median(crop, skin, (230, 190, 160))
    hair_c = _median(crop, hair, (60, 40, 35))
    shirt_c = _shirt_color(photo, box)

    # head on a small grid: measured hair silhouette + idealized round face
    hm = resize.resize_area(hair.astype(np.float32), HEAD_H, HEAD_W) > 0.45
    if symmetric_hair:
        hm = hm | hm[:, ::-1]
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W]
    cx = face.center_x * HEAD_W - 0.5
    sm = ((gx - cx) / (HEAD_W * 0.30)) ** 2 + ((gy - HEAD_H * 0.58) / (HEAD_H * 0.36)) ** 2 < 1
    sm &= ~(hm & (gy < HEAD_H * 0.40))          # fringe stays over the forehead
    hm &= ~sm

    head = np.zeros((HEAD_H, HEAD_W, 3))
    head[hm] = hair_c
    head[sm] = skin_c
    head[sm & neighbor(hm, -1, 0)] = skin_c * 0.82          # shadow under the fringe
    alpha = hm | sm

    # Chibi features sit at fixed places on the round face (template), so the
    # character always looks right; only the face center comes from the photo.
    ey = int(round(HEAD_H * 0.56))
    for ex in (int(round(cx - HEAD_W * 0.17)) - 1, int(round(cx + HEAD_W * 0.17))):
        head[ey - 1:ey + 2, ex:ex + 2] = (40, 28, 30)
        head[ey - 1, ex] = (255, 255, 255)
        alpha[ey - 1:ey + 2, ex:ex + 2] = True
        head[ey + 2, ex:ex + 2] = skin_c * 0.6 + np.array([255, 120, 130]) * 0.4   # blush
    my, mx = int(round(HEAD_H * 0.78)), int(round(cx))
    head[my, mx:mx + 2] = (200, 80, 90)

    body = np.zeros((BODY_H, BODY_W, 3))
    ba = np.zeros((BODY_H, BODY_W), bool)
    def fill(rs, cs, color):
        body[rs, cs] = color
        ba[rs, cs] = True
    fill(slice(0, 6), slice(2, 14), shirt_c)                 # torso
    fill(slice(1, 5), slice(0, 2), shirt_c * 0.85)            # arms
    fill(slice(1, 5), slice(14, 16), shirt_c * 0.85)
    fill(5, slice(0, 2), skin_c)                              # hands
    fill(5, slice(14, 16), skin_c)
    fill(slice(6, 9), slice(4, 7), PANTS)                     # legs
    fill(slice(6, 9), slice(9, 12), PANTS)
    fill(9, slice(3, 7), SHOES)
    fill(9, slice(9, 13), SHOES)
    body[0, 6:10] = shirt_c * 0.7                             # collar

    # assemble: body first, head on top, then a 1-pixel silhouette outline
    w = max(HEAD_W, BODY_W) + 2
    h = HEAD_H + BODY_H + 1
    img = np.zeros((h, w, 3))
    mask = np.zeros((h, w), bool)
    bx, by = (w - BODY_W) // 2, HEAD_H - 2
    img[by:by + BODY_H, bx:bx + BODY_W][ba] = body[ba]
    mask[by:by + BODY_H, bx:bx + BODY_W] |= ba
    hx = (w - HEAD_W) // 2
    img[1:1 + HEAD_H, hx:hx + HEAD_W][alpha] = head[alpha]
    mask[1:1 + HEAD_H, hx:hx + HEAD_W] |= alpha
    ring = np.zeros_like(mask)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ring |= neighbor(mask, dy, dx)
    ring &= ~mask
    img[ring] = hair_c * 0.45
    mask |= ring
    return Character(image=np.clip(np.round(img), 0, 255).astype(np.uint8), mask=mask, face=face)
