"""Character mode: photo -> chibi pixel character -> composited on a pixel background.

    detect.py   find the head box and the eyes / mouth (skin color + projection profiles)
    sprite.py   build the 2-head-tall chibi sprite from the photo
    scene.py    pixel background (photo via Module A, or the built-in Namsan scene) + compositing

OWNER: M1 (Leader).
"""
from .sprite import Character, make_character  # noqa: F401
from .scene import compose_scene, default_background, pixel_background  # noqa: F401
