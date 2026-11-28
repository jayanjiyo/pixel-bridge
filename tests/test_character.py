"""M1: pixelbridge/character (character mode)

    python -m pytest tests/test_character.py
"""
import numpy as np
import pytest

from pixelbridge.character import compose_scene, default_background, make_character
from pixelbridge.character.detect import find_head_box, largest_component, skin_mask

def portrait():
    """Synthetic selfie: gray background, dark hair, skin-colored face, red shirt."""
    h, w = 300, 300
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w, 3), (150, 155, 160), np.uint8)
    img[((xx - 150) / 85) ** 2 + ((yy - 130) / 100) ** 2 < 1] = (45, 30, 25)        # hair
    img[((xx - 150) / 60) ** 2 + ((yy - 140) / 75) ** 2 < 1] = (232, 188, 160)      # face
    img[230:, 60:240] = (200, 40, 50)                                                # shirt
    return img

def test_skin_mask_and_largest_component():
    img = portrait()
    m = skin_mask(img)
    assert m[140, 150] and not m[10, 10] and not m[260, 150]
    two = np.zeros((10, 10), bool)
    two[1:3, 1:3] = True
    two[5:9, 5:9] = True
    big = largest_component(two)
    assert big.sum() == 16 and not big[1, 1]

def test_find_head_box_contains_face():
    x0, y0, x1, y1 = find_head_box(portrait())
    assert x0 < 100 and x1 > 200 and y0 < 80 and y1 > 200

def test_make_character():
    c = make_character(portrait())
    assert c.image.dtype == np.uint8 and c.image.ndim == 3
    assert c.mask.shape == c.image.shape[:2] and c.mask.dtype == bool
    h, w = c.image.shape[:2]
    assert h > w and 30 <= h <= 40                     # 2-head-tall sprite
    lower = c.image[h // 2:][c.mask[h // 2:]].astype(int)
    assert ((lower[:, 0] - lower[:, 1]) > 100).any()   # shirt color comes from the photo (red)

def test_compose_scene_and_shared_palette():
    bg = default_background(128, 96)
    assert bg.shape == (96, 128, 3) and bg.dtype == np.uint8
    c = make_character(portrait())
    scene = compose_scene(bg, c, x=0.3, ground=0.9)
    assert scene.shape == bg.shape and (scene != bg).any()
    pal = np.array([[0, 0, 0], [255, 255, 255], [200, 40, 50], [232, 188, 160], [45, 30, 25]], np.uint8)
    flat = np.zeros_like(bg)
    s2 = compose_scene(flat, c, x=0.5, palette=pal)
    changed = s2[(s2 != 0).any(-1)]
    colors = {tuple(v) for v in changed}
    allowed = {tuple(v) for v in pal} | {tuple(v) for v in c.image.reshape(-1, 3)}
    assert colors <= allowed

def test_character_near_border_is_clipped():
    c = make_character(portrait())
    scene = compose_scene(default_background(64, 48), c, x=0.0, ground=0.3)
    assert scene.shape == (48, 64, 3)

# ------------------------------------------------------------------ attributes and templates
from pixelbridge.character import measure_attributes, render_character  # noqa: E402
from pixelbridge.character.hair import BANGS, HEAD_H, HEAD_W, LENGTHS, face_mask, hair_masks  # noqa: E402
from pixelbridge.character.measure import xdog  # noqa: E402

def test_xdog_finds_a_dark_line():
    g = np.ones((40, 40), np.float32)
    g[20, 5:35] = 0.0
    lines = xdog(g) < 0.5
    assert lines[20, 20] and not lines[5, 5]

def test_measure_attributes_fields():
    img = portrait()
    a = measure_attributes(img, find_head_box(img))
    assert a.bangs in BANGS and a.length in LENGTHS and 0 <= a.part <= 1
    assert isinstance(a.smile, bool) and isinstance(a.happy_eyes, bool)
    assert a.shirt[0] > a.shirt[1] + 60          # red shirt

def test_hair_templates_all_combinations():
    face = face_mask()
    for length in LENGTHS:
        for bangs in BANGS:
            hair, below = hair_masks(length, bangs)
            assert hair.shape == (HEAD_H, HEAD_W) and hair.dtype == bool
            assert (face & ~hair).sum() > 40      # the face stays visible
    full, _ = hair_masks("long", "full")
    none, _ = hair_masks("long", "none")
    assert (full & face).sum() > (none & face).sum()
    assert hair_masks("long", "none")[1].sum() > hair_masks("short", "none")[1].sum() == 0

def test_overrides_change_the_sprite():
    img = portrait()
    base = make_character(img)
    edited = make_character(img, bangs="full", length="short", smile=True, happy_eyes=True)
    assert edited.attrs.bangs == "full" and edited.attrs.length == "short"
    assert edited.image.shape != base.image.shape or (edited.image != base.image).any()
    again = render_character(edited.attrs)        # rendering is deterministic
    assert np.array_equal(again.image, edited.image)

def test_find_head_box_reports_missing_face():
    box, found = find_head_box(portrait(), return_found=True)
    assert found and box == find_head_box(portrait())
    scenery = np.zeros((200, 300, 3), np.uint8)
    scenery[:100] = (120, 170, 230)                      # sky
    scenery[100:] = (90, 140, 70)                        # grass, no skin at all
    assert find_head_box(scenery, return_found=True)[1] is False
    strip = scenery.copy()
    strip[150:160, :] = (232, 188, 160)                  # skin-colored but a thin strip
    assert find_head_box(strip, return_found=True)[1] is False

# ------------------------------------------------------------------ eyes and eye-anchored features
from dataclasses import replace  # noqa: E402

from pixelbridge.character import analyze  # noqa: E402
from pixelbridge.character.eyes import find_eyes  # noqa: E402
from pixelbridge.character.features import measure_from_eyes  # noqa: E402
from pixelbridge.character.hair import FACES, VOLUMES  # noqa: E402

def face_photo(hair=(45, 30, 25), long_hair=False, teeth=False):
    """Synthetic front-facing portrait with eyes, eyebrows, nostrils and a mouth."""
    h, w = 400, 320
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w, 3), (150, 160, 175), np.uint8)                                # background
    img[((xx - 160) / 95) ** 2 + ((yy - 150) / 115) ** 2 < 1] = hair                    # hair
    if long_hair:
        img[150:330, 65:255] = hair
    img[300:, 60:260] = (40, 90, 200)                                                   # blue shirt
    img[250:310, 135:185] = (225, 180, 150)                                             # neck
    img[((xx - 160) / 68) ** 2 + ((yy - 165) / 88) ** 2 < 1] = (232, 188, 160)          # face
    for ex in (128, 192):
        img[((xx - ex) / 11) ** 2 + ((yy - 160) / 6) ** 2 < 1] = (35, 25, 25)           # eyes
        img[138:143, ex - 14:ex + 14] = (60, 40, 35)                                    # eyebrows
    img[196:200, 150:155] = img[196:200, 165:170] = (120, 80, 70)                       # nostrils
    img[222:228, 138:182] = (245, 245, 240) if teeth else (170, 70, 80)                 # mouth
    return img

def test_find_eyes_on_a_face_and_not_on_a_faceless_picture():
    eyes = find_eyes(face_photo())
    assert eyes is not None
    (lx, ly), (rx, ry) = eyes["left"], eyes["right"]
    assert abs(lx - 128) < 8 and abs(rx - 192) < 8 and abs(ly - 160) < 8 and abs(ry - 160) < 8
    assert find_eyes(portrait()) is None                 # no eyes drawn: fall back to the skin box
    assert find_eyes(np.full((200, 200, 3), 128, np.uint8)) is None

def test_features_differ_between_people():
    dark_short = measure_from_eyes(face_photo(), find_eyes(face_photo()))
    blond = face_photo(hair=(225, 190, 120), long_hair=True, teeth=True)
    blond_long = measure_from_eyes(blond, find_eyes(blond))
    assert dark_short.hair.mean() < 80 < blond_long.hair.mean()          # hair color from the photo
    assert dark_short.length in ("crop", "short") and blond_long.length in ("medium", "long")
    assert blond_long.teeth and not dark_short.teeth
    assert dark_short.shirt[2] > dark_short.shirt[0] + 60                # blue shirt
    a, found = analyze(face_photo())
    assert found and "eye_distance" in a.scores                          # the eye-based path was used
    assert (render_character(dark_short).image.shape != render_character(blond_long).image.shape
            or (render_character(dark_short).image != render_character(blond_long).image).any())

def test_new_attributes_change_the_sprite():
    base = make_character(portrait()).attrs
    ref = render_character(base).image
    for change in (dict(face="slim"), dict(face="round"), dict(jaw="pointed"), dict(brows="bold"),
                   dict(eyes="small"), dict(teeth=True), dict(volume="big"), dict(length="crop"),
                   dict(beard="chin"), dict(beard="full"), dict(lip=(120, 40, 90)),
                   dict(brows="bold", brow_color=(200, 170, 110)), dict(eye_color=(60, 90, 140))):
        img = render_character(replace(base, **change)).image
        assert img.shape != ref.shape or (img != ref).any(), change
    for face in FACES:
        for volume in VOLUMES:
            hair, _ = hair_masks("crop", "none", face=face, volume=volume)
            assert (face_mask(face) & ~hair).sum() > 40

def test_hair_is_drawn_from_the_photo():
    from pixelbridge.character.hair import SHAPE_ROWS, photo_hair_masks
    short = measure_from_eyes(face_photo(), find_eyes(face_photo()))
    long_photo = face_photo(hair=(225, 190, 120), long_hair=True)
    long = measure_from_eyes(long_photo, find_eyes(long_photo))
    assert short.hair_shape is not None and short.hair_shape.shape == (SHAPE_ROWS, HEAD_W)
    assert long.hair_shape[HEAD_H:].sum() > short.hair_shape[HEAD_H:].sum()      # hair below the head only when long
    hair, below = photo_hair_masks(long.hair_shape, long.face, long.jaw)
    assert hair.shape == (HEAD_H, HEAD_W) and below.shape == (SHAPE_ROWS - HEAD_H, HEAD_W)
    face = face_mask(long.face, long.jaw)
    assert (face & ~hair).sum() > 40                                              # the face stays visible
    assert not (hair & face)[int(HEAD_H * 0.56):].any()                           # never over the eyes or mouth
    # above the eyes the outline is a dome: left-right symmetric even when the photo hair is
    # lopsided, as high as the hair in the photo; specks below are dropped
    lopsided = np.zeros((SHAPE_ROWS, HEAD_W), bool)
    lopsided[1:8, 2:12] = True
    lopsided[28, 20] = True
    hair, below = photo_hair_masks(lopsided)
    ey = int(round(HEAD_H * 0.56))
    outside = hair & ~face_mask()                                                 # bangs are not part of the dome
    assert np.array_equal(outside[:ey - 2], outside[:ey - 2, ::-1]) and not below.any()
    assert hair[1].any() and not hair[0].any()                                    # top of the dome = top of the hair
    low = np.zeros((SHAPE_ROWS, HEAD_W), bool)
    low[4:8, 4:18] = True
    assert not photo_hair_masks(low)[0][:4].any()                                 # flatter hair, lower dome
    with_shape = render_character(short).image
    templated = render_character(replace(short, hair_shape=None)).image
    assert with_shape.shape != templated.shape or (with_shape != templated).any()
    empty, _ = photo_hair_masks(np.zeros((SHAPE_ROWS, HEAD_W), bool))             # no hair at all is allowed
    assert not empty.any()

def test_find_eyes_by_layout():
    """The layout search is only a second try (see sprite.analyze): it must give
    eyes in the right form, in left-to-right order, and nothing on a flat image."""
    from pixelbridge.character.eyes import find_eyes_by_layout
    assert find_eyes_by_layout(np.full((200, 200, 3), 128, np.uint8)) is None
    eyes = find_eyes_by_layout(face_photo())
    if eyes is not None:
        assert eyes["left"][0] < eyes["right"][0] and eyes["score"] > 0
    a, found = analyze(face_photo())                     # the first method wins when both find something
    assert found and abs(a.scores["eye_distance"] - 64) < 16

def test_tilted_head_is_found_and_measured_upright():
    from pixelbridge.character.eyes import find_eyes_tilted, rotate, rotate_point, upright
    img = face_photo()
    c = ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2)
    assert np.array_equal(rotate(img, 0.0), img)
    x, y = rotate_point(rotate_point((100.0, 50.0), 20.0, c), -20.0, c)       # there and back
    assert abs(x - 100) < 1e-6 and abs(y - 50) < 1e-6
    tilted = rotate(img, 20.0)                                                # head tilted by 20 degrees
    eyes = find_eyes_tilted(tilted)
    assert eyes is not None
    (lx, ly), (rx, ry) = eyes["left"], eyes["right"]
    assert abs(np.hypot(rx - lx, ry - ly) - 64) < 12                          # same eye distance as upright
    assert abs(abs(np.degrees(np.arctan2(ry - ly, rx - lx))) - 20) < 8        # the eye line is tilted
    straight, level = upright(tilted, eyes)
    assert abs(level["left"][1] - level["right"][1]) < 2                      # level after straightening
    a, found = analyze(tilted)
    assert found and a.hair.mean() < 80 and a.shirt[2] > a.shirt[0] + 60      # still dark hair, blue shirt

def test_photo_hair_rules():
    from pixelbridge.character.hair import SHAPE_ROWS, photo_hair_masks
    ey = int(round(HEAD_H * 0.56))
    shape = np.zeros((SHAPE_ROWS, HEAD_W), bool)
    shape[1:6, 4:18] = True                      # hair on top
    shape[6:28, 2:4] = True                      # long hair on the left side only
    hair, below = photo_hair_masks(shape, length="long")
    assert below[:, :HEAD_W // 2].any() and below[:, HEAD_W // 2:].any()        # mirrored to the right
    hair, below = photo_hair_masks(shape, length="crop")
    assert not below.any() and not hair[ey + 2:].any()                          # nothing below the ears
    hair, below = photo_hair_masks(shape, length="short")
    assert not below[:, HEAD_W // 2:].any()                                     # short hair is not mirrored
    face = face_mask()
    top_only = np.zeros((SHAPE_ROWS, HEAD_W), bool)
    top_only[1:6, 6:16] = True
    hair, _ = photo_hair_masks(top_only, length="crop")
    side = np.nonzero(face[ey])[0]
    assert hair[ey, side[0] - 1] and hair[ey, side[-1] + 1]                     # hair reaches the temples
    blob = top_only.copy()
    blob[16:20, 0:3] = True                      # a hand or a couch of the same color, not connected
    hair, _ = photo_hair_masks(blob, length="short")
    assert not hair[16:20, 0:3].any()

# ------------------------------------------------------------------ built-in landmark backgrounds
from pixelbridge.character import LANDMARKS, landmark_background  # noqa: E402

def test_landmarks_have_size_and_ground():
    assert len(LANDMARKS) == 5
    for key in LANDMARKS:
        name, bg = landmark_background(key)
        assert name == key and bg.shape == (96, 128, 3) and bg.dtype == np.uint8
        assert len(np.unique(bg[:80].reshape(-1, 3), axis=0)) > 8      # an actual picture, not a flat fill
        assert np.array_equal(bg, landmark_background(key)[1])           # drawn the same every time

def test_random_landmark_and_compose():
    rng = np.random.default_rng(0)
    seen = {landmark_background("random", rng)[0] for _ in range(40)}
    assert seen == set(LANDMARKS)
    c = make_character(portrait())
    name, bg = landmark_background("hangang")
    scene = compose_scene(bg, c, x=0.3, ground=0.92)
    assert scene.shape == bg.shape and (scene != bg).any()
    with pytest.raises(ValueError):
        landmark_background("eiffel")
