"""Build a 2-head-tall (chibi) pixel character from a photo.

Measured from the photo (measure.py): expression (smile, smiling eyes), hair
(bangs, part, length, color), skin and shirt colors, light direction.
Drawn from templates: round face, eyes, blush, hair shapes (hair.py), small body.
Every measured attribute can be overridden (CLI / GUI).

Pixel-art conventions used: 1-pixel silhouette outline only (never inside the
face), shadows that are darker AND shifted toward red/purple (hue shifting),
a small highlight on the lit side of the hair.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.color import lab_to_rgb, rgb_to_lab
from ..core.filters import neighbor
from .detect import find_head_box
from .hair import EYE_ROW, HEAD_H, HEAD_W, face_mask, hair_masks
from .measure import Attributes, measure_attributes

BODY_W, BODY_H = 16, 10
PANTS = (60, 70, 110)
SHOES = (40, 40, 45)
EYE = (40, 28, 30)
MOUTH = (190, 75, 85)
BELOW_ROWS = 7


@dataclass
class Character:
    image: np.ndarray        # (h, w, 3) uint8
    mask: np.ndarray         # (h, w) bool, True where the character is
    attrs: Attributes


def _shade(color, dL: float) -> np.ndarray:
    """Darker (dL < 0) with a hue shift toward red/purple, or lighter (dL > 0) toward yellow."""
    lab = rgb_to_lab(np.clip(np.asarray(color, float), 0, 255).astype(np.uint8)[None])[0].astype(np.float64)
    lab[0] += dL
    if dL < 0:
        lab[1] -= dL * 0.25
        lab[2] += dL * 0.35
    else:
        lab[2] += dL * 0.2
    return np.clip(lab_to_rgb(lab[None])[0] * 255, 0, 255)


def _head(attrs: Attributes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hair, below = hair_masks(attrs.length, attrs.bangs, attrs.part, BELOW_ROWS)
    face = face_mask()
    skin = face & ~hair
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W]
    cx = HEAD_W * 0.5 - 0.5
    sk, hc = np.asarray(attrs.skin, float), np.asarray(attrs.hair, float)

    head = np.zeros((HEAD_H, HEAD_W, 3))
    head[hair] = hc
    head[hair & (gy > HEAD_H * 0.62)] = _shade(hc, -5)                 # darker underside
    head[skin] = sk
    head[skin & neighbor(hair, -1, 0)] = _shade(sk, -9)                # shadow under the hair
    # rim light: hair pixels on the top edge, on the lit side, one shade lighter
    lit = 1 if attrs.shadow_side == "left" else -1
    rim = hair & ~neighbor(hair, -1, 0) & ((gx - cx) * lit > 0) & (gy < HEAD_H * 0.35)
    head[rim | (neighbor(rim, 0, -lit) & hair & (gy < HEAD_H * 0.2))] = _shade(hc, 10)

    ey = int(round(HEAD_H * EYE_ROW))
    for ex in (int(round(cx - HEAD_W * 0.17)) - 1, int(round(cx + HEAD_W * 0.17))):
        if attrs.happy_eyes:                                           # smiling eyes ^ ^
            head[ey, ex] = EYE
            head[ey - 1, ex + 1] = EYE
            head[ey, ex + 2] = EYE
        else:
            head[ey - 1:ey + 2, ex:ex + 2] = EYE
            head[ey - 1, ex] = (255, 255, 255)
        head[ey + 2, ex:ex + 2] = sk * 0.55 + np.array([255, 120, 130]) * 0.45   # blush
    my, mx = int(round(HEAD_H * 0.78)), int(round(cx))
    head[my, mx:mx + 2] = MOUTH
    if attrs.smile:
        head[my - 1, mx - 1] = MOUTH
        head[my - 1, mx + 2] = MOUTH
    return head, hair | skin, below


def render_character(attrs: Attributes) -> Character:
    """Draw the sprite from attributes (measured or edited)."""
    head, alpha, below = _head(attrs)
    sk, hc, sh = (np.asarray(c, float) for c in (attrs.skin, attrs.hair, attrs.shirt))

    body = np.zeros((BODY_H, BODY_W, 3))
    ba = np.zeros((BODY_H, BODY_W), bool)
    def fill(rs, cs, color):
        body[rs, cs] = color
        ba[rs, cs] = True
    fill(slice(0, 6), slice(2, 14), sh)                       # torso
    fill(slice(0, 6), slice(2, 4), _shade(sh, -8))            # side shading
    fill(slice(1, 5), slice(0, 2), _shade(sh, -10))           # arms
    fill(slice(1, 5), slice(14, 16), _shade(sh, -10))
    fill(5, slice(0, 2), sk)                                  # hands
    fill(5, slice(14, 16), sk)
    fill(slice(6, 9), slice(4, 7), PANTS)                     # legs
    fill(slice(6, 9), slice(9, 12), PANTS)
    fill(9, slice(3, 7), SHOES)
    fill(9, slice(9, 13), SHOES)

    w = HEAD_W + 2
    h = 1 + HEAD_H + BELOW_ROWS + 1
    img = np.zeros((h, w, 3))
    mask = np.zeros((h, w), bool)
    hx = (w - HEAD_W) // 2
    # back to front: long hair behind the body, body, head
    rows = slice(1 + HEAD_H, 1 + HEAD_H + BELOW_ROWS)
    img[rows, hx:hx + HEAD_W][below] = _shade(hc, -5)
    mask[rows, hx:hx + HEAD_W] |= below
    bx, by = (w - BODY_W) // 2, HEAD_H - 2
    img[by:by + BODY_H, bx:bx + BODY_W][ba] = body[ba]
    mask[by:by + BODY_H, bx:bx + BODY_W] |= ba
    img[1:1 + HEAD_H, hx:hx + HEAD_W][alpha] = head[alpha]
    mask[1:1 + HEAD_H, hx:hx + HEAD_W] |= alpha

    ring = np.zeros_like(mask)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ring |= neighbor(mask, dy, dx)
    ring &= ~mask
    img[ring] = hc * 0.45
    mask |= ring
    last = np.nonzero(mask.any(axis=1))[0].max() + 1             # trim empty rows at the bottom
    return Character(image=np.clip(np.round(img[:last]), 0, 255).astype(np.uint8),
                     mask=mask[:last], attrs=attrs)


def make_character(photo: np.ndarray, head_box: tuple[int, int, int, int] | None = None,
                   **overrides) -> Character:
    """photo (H, W, 3) uint8 -> Character.

    overrides: any Attributes field, e.g. bangs="full", length="short",
    smile=True, happy_eyes=False, part=0.4, hair=(40, 30, 30).
    """
    attrs = measure_attributes(photo, head_box or find_head_box(photo))
    if overrides:
        attrs = replace(attrs, **{k: v for k, v in overrides.items() if v is not None})
    return render_character(attrs)
