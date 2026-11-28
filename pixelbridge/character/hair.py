"""Hair templates for the chibi head.

A hairstyle is chosen from a few parameters (length x bangs x part) instead of
copying the measured outline, which is noisy at 22 pixels (background and
shadows leak into it). The parameters come from measure.py and can be
overridden by the user.

    length: "short" (bob, to the chin) | "medium" (shoulder) | "long"
    bangs:  "none" | "curtain" (parted in the middle) | "full" | "side" (swept)
"""
from __future__ import annotations

import numpy as np

HEAD_W, HEAD_H = 22, 23
LENGTHS = ("short", "medium", "long")
BANGS = ("none", "curtain", "full", "side")
EYE_ROW = 0.56            # eye row on the head grid (fraction of HEAD_H)


def face_mask() -> np.ndarray:
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W]
    cx = HEAD_W * 0.5 - 0.5
    return ((gx - cx) / (HEAD_W * 0.30)) ** 2 + ((gy - HEAD_H * 0.58) / (HEAD_H * 0.36)) ** 2 < 1


def hair_masks(length: str, bangs: str, part: float = 0.5, extra_rows: int = 10):
    """Return (hair, below) masks.

    hair  : (HEAD_H, HEAD_W) hair on the head grid, fringe included
    below : (extra_rows, HEAD_W) hair that continues below the head (drawn behind the body)
    """
    if length not in LENGTHS or bangs not in BANGS:
        raise ValueError(f"length must be one of {LENGTHS}, bangs one of {BANGS}")
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W].astype(float)
    cx = HEAD_W * 0.5 - 0.5
    face = face_mask()
    rx = HEAD_W * 0.46
    cap = ((gx - cx) / rx) ** 2 + ((gy - HEAD_H * 0.46) / (HEAD_H * 0.44)) ** 2 < 1
    bottom = {"short": HEAD_H * 0.80, "medium": HEAD_H, "long": HEAD_H}[length]
    sides = (np.abs(gx - cx) < rx) & (np.abs(gx - cx) > HEAD_W * 0.24) & (gy > HEAD_H * 0.40) & (gy < bottom)
    if length == "short":                                  # bob tucks inward at the bottom
        sides &= ~((gy > HEAD_H * 0.70) & (np.abs(gx - cx) > rx - 1.5))
    hair = (cap | sides) & ~face

    pc = part * HEAD_W - 0.5
    ey = int(round(HEAD_H * EYE_ROW))
    if bangs == "none":                                    # forehead visible, hairline peaks at the part
        fringe = face & (gy < np.minimum(HEAD_H * 0.27 + 0.30 * np.abs(gx - pc), HEAD_H * 0.40))
    elif bangs == "full":
        fringe = face & (gy <= ey - 2)
    elif bangs == "curtain":                               # parted bangs: gap widens downward
        fringe = face & (gy < HEAD_H * 0.36)
        y = gy >= HEAD_H * 0.36
        fringe |= face & y & (gy <= ey - 2) & (np.abs(gx - pc) > (gy - HEAD_H * 0.36) * 0.8 + 0.5)
    else:                                                  # side-swept: diagonal edge from the part
        toward = -1 if part < 0.5 else 1                   # hair falls away from the part
        edge = HEAD_H * 0.30 + toward * (gx - pc) * 1.0
        fringe = face & ((gy < HEAD_H * 0.30) | ((gy < edge) & (gy <= ey - 2)))

    gyb, gxb = np.mgrid[:extra_rows, :HEAD_W].astype(float)
    n = {"short": 0, "medium": 3, "long": extra_rows}[length]
    below = (gyb < n) & (np.abs(gxb - cx) > HEAD_W * 0.24) & (np.abs(gxb - cx) < rx - gyb * 0.08)
    return hair | fringe, below
