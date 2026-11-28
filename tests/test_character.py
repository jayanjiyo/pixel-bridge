"""M1: pixelbridge/character (character mode)

    python -m pytest tests/test_character.py
"""
import numpy as np

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
