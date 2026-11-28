"""Character mode: photo -> chibi pixel character -> composited on a pixel background.

    detect.py   find the head box and the eyes / mouth (skin color + projection profiles)
    measure.py  measure expression, hair, colors and light from the photo (XDoG lines)
    hair.py     hair templates: length x bangs x part
    sprite.py   draw the 2-head-tall chibi sprite from the measured (or edited) attributes
    scene.py    pixel background (photo via Module A, or the built-in Namsan scene) + compositing

OWNER: M1 (Leader).
"""
from .measure import Attributes, measure_attributes  # noqa: F401
from .sprite import Character, make_character, render_character  # noqa: F401
from .scene import compose_scene, default_background, pixel_background  # noqa: F401
