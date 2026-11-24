"""Module B: pixel art -> high resolution.

    similarity.py  M4   YUV color distance (shared by the rule-based scalers)
    scalex.py      M4   Scale2x, Scale3x, Eagle
    xbr.py         M4   xBR (levels 1 and 2)
    slope.py       M4   OUR extended slope method ("ours")
    pipeline.py    M1   method dispatch (already implemented);
                        nearest / bilinear / bicubic baselines live in core/resize.py (M1)
"""
from .pipeline import METHODS, SUPPORTED_SCALES, upscale, upscale_all  # noqa: F401
