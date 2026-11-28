"""Character mode: photo -> chibi pixel character -> composited on a pixel background.

    detect.py   find the head box and the eyes / mouth (skin color + projection profiles)
    sprite.py   build the 2-head-tall chibi sprite from the photo

OWNER: M1 (Leader).
"""
from .sprite import Character, make_character  # noqa: F401
