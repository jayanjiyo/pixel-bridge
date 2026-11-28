"""Hair templates for the chibi head.

A hairstyle is chosen from a few parameters (length x bangs x part) instead of
copying the measured outline, which is noisy at 22 pixels (background and
shadows leak into it). The parameters come from measure.py and can be
overridden by the user.

    length: "crop" (cut above the ears) | "short" (bob, to the chin) | "medium" (shoulder) | "long"
    bangs:  "none" | "curtain" (parted in the middle) | "full" | "side" (swept)
"""
from __future__ import annotations

import numpy as np

HEAD_W, HEAD_H = 22, 23
LENGTHS = ("crop", "short", "medium", "long")
BANGS = ("none", "curtain", "full", "side")
EYE_ROW = 0.56            # eye row on the head grid (fraction of HEAD_H)
FACES = {"slim": 0.265, "oval": 0.30, "round": 0.335}      # half width of the face / HEAD_W
JAWS = ("round", "pointed")
VOLUMES = {"flat": (0.44, 0.40), "normal": (0.46, 0.44), "big": (0.49, 0.48)}   # hair cap radii (x, y)
DOME_POWER = 3.0          # roundness of the top of the hair (2 = ellipse, higher = flatter top)
BELOW_ROWS = 7            # rows of long hair drawn below the head, behind the body
SHAPE_ROWS = HEAD_H + BELOW_ROWS      # height of a hair silhouette taken from the photo


def face_mask(face: str = "oval", jaw: str = "round") -> np.ndarray:
    """Face shape on the head grid. The lower half follows a narrower curve for a
    pointed jaw (superellipse exponent 1.45 instead of 2)."""
    if face not in FACES or jaw not in JAWS:
        raise ValueError(f"face must be one of {tuple(FACES)}, jaw one of {JAWS}")
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W]
    cx = HEAD_W * 0.5 - 0.5
    dx = np.abs(gx - cx) / (HEAD_W * FACES[face])
    dy = (gy - HEAD_H * 0.58) / (HEAD_H * 0.36)
    p = np.where((dy > 0) & (jaw == "pointed"), 1.45, 2.0)
    return dx ** p + np.abs(dy) ** p < 1


def hair_masks(length: str, bangs: str, part: float = 0.5, extra_rows: int = 10,
               face: str = "oval", jaw: str = "round", volume: str = "normal"):
    """Return (hair, below) masks.

    hair  : (HEAD_H, HEAD_W) hair on the head grid, fringe included
    below : (extra_rows, HEAD_W) hair that continues below the head (drawn behind the body)
    """
    if length not in LENGTHS or bangs not in BANGS or volume not in VOLUMES:
        raise ValueError(f"length must be one of {LENGTHS}, bangs one of {BANGS}, volume one of {tuple(VOLUMES)}")
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W].astype(float)
    cx = HEAD_W * 0.5 - 0.5
    face = face_mask(face, jaw)
    rx = HEAD_W * VOLUMES[volume][0]
    cap = ((gx - cx) / rx) ** 2 + ((gy - HEAD_H * 0.46) / (HEAD_H * VOLUMES[volume][1])) ** 2 < 1
    bottom = {"crop": HEAD_H * 0.56, "short": HEAD_H * 0.80, "medium": HEAD_H, "long": HEAD_H}[length]
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
    n = {"crop": 0, "short": 0, "medium": 3, "long": extra_rows}[length]
    below = (gyb < n) & (np.abs(gxb - cx) > HEAD_W * 0.24) & (np.abs(gxb - cx) < rx - gyb * 0.08)
    return hair | fringe, below


def _round_top(hair: np.ndarray, fmask: np.ndarray, ey: int) -> np.ndarray:
    """Replace the hair above the eyes by a rounded dome of the same width and height.

    Width: the widest the hair gets above the eyes, at least one cell beyond the face.
    Height: the top of the hair over the middle of the head. Hair inside the face (bangs)
    and hair below the eyes are kept as they are.
    """
    upper = hair[:ey + 1] & ~fmask[:ey + 1]
    cols = np.nonzero(upper.any(axis=0))[0]
    if not len(cols):
        return hair
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W]
    cx = HEAD_W * 0.5 - 0.5
    face_cols = np.nonzero(fmask.any(axis=0))[0]
    rx = max(cx - cols[0], cols[-1] - cx, (face_cols[-1] - face_cols[0]) / 2 + 1.0) + 0.5
    rx = min(rx, HEAD_W / 2)
    middle = upper[:, int(cx) - 2:int(cx) + 4]
    rows = np.nonzero(middle.any(axis=1))[0] if middle.any() else np.nonzero(upper.any(axis=1))[0]
    ry = ey - rows[0] + 0.5
    # a superellipse: flatter on top than an ellipse, which looks pointed when the hair is high
    dome = (np.abs((gx - cx) / rx) ** DOME_POWER + np.abs((gy - ey) / ry) ** DOME_POWER <= 1) & (gy <= ey)
    out = hair.copy()
    out[:ey + 1] = (hair[:ey + 1] & fmask[:ey + 1]) | (dome[:ey + 1] & ~fmask[:ey + 1])
    return out


def photo_hair_masks(shape: np.ndarray, face: str = "oval", jaw: str = "round", length: str | None = None):
    """Hair masks from the silhouette measured in the photo (features.hair_grid),
    in the same form as hair_masks: (hair on the head grid, hair below the head).

    No template is used: the photo decides the whole outline (volume, asymmetry,
    a side part, curls sticking out, bangs, long hair, no hair at all). Only what
    keeps the sprite readable is enforced, in this order:
    1. inside the face, hair is allowed on the forehead only (bangs), so the eyes
       and the mouth are never covered;
    2. empty cells between the hair and the top of the face are filled (no holes);
       above the eyes the outline is rounded: a dome as wide and as high as the hair
       in the photo (a silhouette copied cell by cell looks jagged on 22 pixels);
    3. a cropped cut (length "crop") has no hair below the ears, and neither has short
       hair that hangs down on one side only (a turned head);
    4. long hair (length "medium" or "long") found on one side only is mirrored to the other side;
    5. only the largest connected part is kept (specks, a hand or a couch of the
       same color are dropped);
    6. the hair reaches down to the temples: the cells that touch the face, from the
       top down to the eyes, are hair (very short sides look like skin in a photo).
    """
    shape = np.asarray(shape, bool)
    if shape.shape != (SHAPE_ROWS, HEAD_W):
        raise ValueError(f"hair shape must be {(SHAPE_ROWS, HEAD_W)}, got {shape.shape}")
    gy, _ = np.mgrid[:HEAD_H, :HEAD_W]
    fmask = face_mask(face, jaw)
    ey = int(round(HEAD_H * EYE_ROW))
    head = shape[:HEAD_H]
    hair = (head & ~fmask) | (head & fmask & (gy <= ey - 2))                    # 1
    for x in range(HEAD_W):                                                     # 2
        rows_hair = np.nonzero(hair[:, x])[0]
        rows_face = np.nonzero(fmask[:, x])[0]
        if len(rows_hair) and len(rows_face) and rows_hair[0] < rows_face[0]:
            hair[rows_hair[0]:rows_face[0], x] = True
    hair = _round_top(hair, fmask, ey)                                          # 2b
    full = np.vstack([hair, shape[HEAD_H:]])
    if length == "crop":                                                        # 3
        full[ey + 2:] = False
    lower = full.copy()                                                         # 4
    lower[:ey + 2] = False
    lower[:HEAD_H] &= ~fmask
    left, right = int(lower[:, :HEAD_W // 2].sum()), int(lower[:, HEAD_W // 2:].sum())
    one_sided = min(left, right) < 0.5 * max(left, right)
    if length == "short" and one_sided:
        # short hair that hangs down on one side only is a turned head (the hair above one
        # ear) or a piece of background: a bob has hair on both sides
        full[ey + 2:] = False
    if length in ("medium", "long") and max(left, right) >= 8 and one_sided:
        side = full.copy()                       # the whole side, from the top of the face down,
        side[:int(np.nonzero(fmask.any(axis=1))[0][0])] = False      # so that it joins the hair on top
        mirrored = side[:, ::-1].copy()
        mirrored[:HEAD_H] &= ~fmask
        full |= mirrored
    labels = np.where(full, np.arange(full.size).reshape(full.shape), -1)       # 5 (8-connected)
    for _ in range(full.size):
        grown = labels.copy()
        grown[1:] = np.maximum(grown[1:], labels[:-1])
        grown[:-1] = np.maximum(grown[:-1], labels[1:])
        grown[:, 1:] = np.maximum(grown[:, 1:], labels[:, :-1])
        grown[:, :-1] = np.maximum(grown[:, :-1], labels[:, 1:])
        grown[1:, 1:] = np.maximum(grown[1:, 1:], labels[:-1, :-1])
        grown[1:, :-1] = np.maximum(grown[1:, :-1], labels[:-1, 1:])
        grown[:-1, 1:] = np.maximum(grown[:-1, 1:], labels[1:, :-1])
        grown[:-1, :-1] = np.maximum(grown[:-1, :-1], labels[1:, 1:])
        grown[~full] = -1
        if np.array_equal(grown, labels):
            break
        labels = grown
    if full.any():
        ids, counts = np.unique(labels[full], return_counts=True)
        full = labels == ids[np.argmax(counts)]
        touch = np.zeros_like(fmask)                                            # 6
        touch[1:] |= fmask[:-1]
        touch[:-1] |= fmask[1:]
        touch[:, 1:] |= fmask[:, :-1]
        touch[:, :-1] |= fmask[:, 1:]
        full[:HEAD_H] |= touch & ~fmask & (gy <= ey)
    return full[:HEAD_H], full[HEAD_H:]
