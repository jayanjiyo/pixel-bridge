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
from .eyes import find_eyes_by_layout, find_eyes_tilted, upright
from .features import measure_from_eyes
from .hair import EYE_ROW, HEAD_H, HEAD_W, face_mask, hair_masks, photo_hair_masks
from .measure import Attributes, measure_attributes

BODY_W, BODY_H = 16, 10
PANTS = (60, 70, 110)
SHOES = (40, 40, 45)
EYE = (40, 28, 30)
MOUTH = (190, 75, 85)
TEETH = (246, 244, 236)
EYE_MAX = 85                 # brightest channel allowed for the eye color (darker = readable on any skin)
STRONG_EYES = 2.4            # eye score that is trusted even when no skin-colored face is found
FACE_WIDTH_RANGE = (1.5, 3.2)   # a real face is this many eye distances wide; outside: the "eyes" were wrong
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


def _head(attrs: Attributes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Head colors, head mask, mask of the hair below the head, colors of that hair."""
    from_photo = attrs.hair_shape is not None
    if from_photo:                                                    # outline taken from the photo
        hair, below = photo_hair_masks(attrs.hair_shape, attrs.face, attrs.jaw, attrs.length)
    else:                                                             # template: length x bangs x volume
        hair, below = hair_masks(attrs.length, attrs.bangs, attrs.part, BELOW_ROWS,
                                 face=attrs.face, jaw=attrs.jaw, volume=attrs.volume)
    face = face_mask(attrs.face, attrs.jaw)
    skin = face & ~hair
    gy, gx = np.mgrid[:HEAD_H, :HEAD_W]
    cx = HEAD_W * 0.5 - 0.5
    sk, hc = np.asarray(attrs.skin, float), np.asarray(attrs.hair, float)

    head = np.zeros((HEAD_H, HEAD_W, 3))
    head[hair] = hc
    below_rgb = np.zeros((BELOW_ROWS, HEAD_W, 3))
    below_rgb[:] = _shade(hc, -5)
    if from_photo and attrs.hair_tone is not None:                     # strands as they are in the photo
        tone = np.asarray(attrs.hair_tone)
        light, dark = _shade(hc, 12), _shade(hc, -11)
        if attrs.hair_colors is not None:                              # real strand colors from the photo
            dark = dark if attrs.hair_colors[0] is None else np.asarray(attrs.hair_colors[0], float)
            light = light if attrs.hair_colors[1] is None else np.asarray(attrs.hair_colors[1], float)
        head[hair & (tone[:HEAD_H] > 0)] = light
        head[hair & (tone[:HEAD_H] < 0)] = dark
        below_rgb[:] = hc
        below_rgb[tone[HEAD_H:] > 0] = light
        below_rgb[tone[HEAD_H:] < 0] = dark
    else:
        head[hair & (gy > HEAD_H * 0.62)] = _shade(hc, -5)             # darker underside
    head[skin] = sk
    head[skin & neighbor(hair, -1, 0)] = _shade(sk, -9)                # shadow under the hair
    if not (from_photo and attrs.hair_tone is not None):
        # rim light: hair pixels on the top edge, on the lit side, one shade lighter
        lit = 1 if attrs.shadow_side == "left" else -1
        rim = hair & ~neighbor(hair, -1, 0) & ((gx - cx) * lit > 0) & (gy < HEAD_H * 0.35)
        head[rim | (neighbor(rim, 0, -lit) & hair & (gy < HEAD_H * 0.2))] = _shade(hc, 10)

    ey = int(round(HEAD_H * EYE_ROW))
    eye = np.asarray(EYE, float)
    if attrs.eye_color is not None:                                    # eye color read at the pupils
        eye = np.asarray(attrs.eye_color, float)
        if eye.max() > EYE_MAX:                                        # keep the hue, but dark enough to read as an eye
            eye = eye * (EYE_MAX / eye.max())
    my, mx = int(round(HEAD_H * 0.78)), int(round(cx))
    if attrs.beard != "none":                                          # facial hair, drawn under the mouth
        bc = np.asarray(_shade(hc, -4)) * 0.75 + sk * 0.25
        if attrs.beard in ("chin", "full"):
            width = 3.5 if attrs.beard == "chin" else HEAD_W
            head[skin & (gy >= my + 1) & (np.abs(gx - cx) <= width)] = bc
        if attrs.beard in ("mustache", "full"):
            head[my - 1, mx - 1:mx + 3][skin[my - 1, mx - 1:mx + 3]] = bc
    if attrs.brows == "none":
        brow_color = None
    elif attrs.brow_color is not None:                                 # color of the eyebrows in the photo
        own = np.asarray(attrs.brow_color, float)
        brow_color = own if attrs.brows == "bold" else own * 0.6 + sk * 0.4
    else:
        brow_color = {"thin": sk * 0.45 + hc * 0.55, "bold": hc * 0.7}[attrs.brows]
    lip = MOUTH if attrs.lip is None else np.asarray(attrs.lip, float)
    for side, ex in ((-1, int(round(cx - HEAD_W * 0.17)) - 1), (1, int(round(cx + HEAD_W * 0.17)))):
        if attrs.happy_eyes:                                           # smiling eyes ^ ^
            head[ey, ex] = eye
            head[ey - 1, ex + 1] = eye
            head[ey, ex + 2] = eye
            eye_top = ey - 1
        elif attrs.eyes == "small":                                    # 2 x 2 instead of 2 x 3
            head[ey:ey + 2, ex:ex + 2] = eye
            head[ey, ex] = (255, 255, 255)
            eye_top = ey
        else:
            head[ey - 1:ey + 2, ex:ex + 2] = eye
            head[ey - 1, ex] = (255, 255, 255)
            eye_top = ey - 1
        head[ey + 2, ex:ex + 2] = sk * 0.55 + np.array([255, 120, 130]) * 0.45   # blush
        if brow_color is not None:                                     # eyebrows, only on visible skin
            by = eye_top - 2
            bx = (ex, ex + 2) if attrs.brows == "thin" else ((ex - 1, ex + 2) if side < 0 else (ex, ex + 3))
            for x in range(*bx):
                if skin[by, x]:
                    head[by, x] = brow_color
    if attrs.teeth:                                                    # open smile showing teeth
        head[my, mx - 1:mx + 3] = TEETH
        head[my, mx - 2] = head[my, mx + 3] = lip
        head[my + 1, mx:mx + 2] = lip
    else:
        head[my, mx:mx + 2] = lip
        if attrs.smile:
            head[my - 1, mx - 1] = lip
            head[my - 1, mx + 2] = lip
    return head, hair | skin, below, below_rgb


def render_character(attrs: Attributes) -> Character:
    """Draw the sprite from attributes (measured or edited)."""
    head, alpha, below, below_rgb = _head(attrs)
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
    img[rows, hx:hx + HEAD_W][below] = below_rgb[below]
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


def analyze(photo: np.ndarray) -> tuple[Attributes, bool]:
    """Measure the attributes of a portrait. Returns (attributes, face found).

    If the eyes are found (eyes.find_eyes, then eyes.find_eyes_by_layout as a
    second try), everything is measured relative to
    them (features.measure_from_eyes): this is what makes two people come out
    as two different characters. The eyes are trusted when a skin-colored face
    is found as well, or when their score is very high (a photo with a color
    cast has no "skin-colored" pixels), and only if the face measured from them
    has a plausible width. Otherwise the older skin-box method is used, and
    face found is False when that region does not look like a face.
    """
    box, found = find_head_box(photo, return_found=True)

    def plausible(eyes) -> Attributes | None:
        attrs = measure_from_eyes(*upright(photo, eyes))    # measure with the eye line level
        # a pair that was something else (a collar, a pattern) gives a face of impossible width
        return attrs if FACE_WIDTH_RANGE[0] <= attrs.scores["face_w"] <= FACE_WIDTH_RANGE[1] else None

    eyes = find_eyes_tilted(photo)                          # 1. dark blobs + symmetry + structure
    if eyes is not None and (found or eyes["score"] >= STRONG_EYES):
        attrs = plausible(eyes)
        if attrs is not None:
            return attrs, True
    if found:                                               # 2. the layout of a face (weak or light eyes)
        eyes = find_eyes_by_layout(photo)
        if eyes is not None:
            # this search is less reliable (it can lock onto the hairline), so it is not
            # repeated on rotated copies, and it must agree with the skin box: the eyes lie inside it
            (lx, ly), (rx, ry) = eyes["left"], eyes["right"]
            if box[0] <= (lx + rx) / 2 <= box[2] and box[1] <= (ly + ry) / 2 <= box[3]:
                attrs = plausible(eyes)
                if attrs is not None:
                    return attrs, True
    return measure_attributes(photo, box), found            # 3. the skin box


def make_character(photo: np.ndarray, head_box: tuple[int, int, int, int] | None = None,
                   **overrides) -> Character:
    """photo (H, W, 3) uint8 -> Character.

    head_box: use this box with the skin-box method instead of finding the face.
    overrides: any Attributes field, e.g. bangs="full", length="short",
    smile=True, happy_eyes=False, part=0.4, hair=(40, 30, 30), face="round".
    """
    attrs = measure_attributes(photo, head_box) if head_box else analyze(photo)[0]
    if overrides:
        attrs = replace(attrs, **{k: v for k, v in overrides.items() if v is not None})
    return render_character(attrs)
