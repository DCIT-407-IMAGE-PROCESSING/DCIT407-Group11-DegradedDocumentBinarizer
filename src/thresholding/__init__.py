"""Thresholding methods and the method registry.

THE CONTRACT (every method follows it)
--------------------------------------
    binarize(gray: np.ndarray, **params) -> np.ndarray

- Input:  2D uint8 grayscale image.
- Output: 2D uint8 binary image, 0 = text (black), 255 = background (white).
- A pixel darker than its threshold becomes 0 (text).

To add a new method: create a new file with a `binarize` function and add one
line to METHODS below.
"""
from .global_threshold import binarize as fixed
from .niblack import binarize as niblack
from .opencv_baselines import (
    binarize_adaptive_gaussian as cv2_adaptive_gaussian,
    binarize_adaptive_mean as cv2_adaptive_mean,
    binarize_otsu as cv2_otsu,
)
from .otsu import binarize as otsu
from .sauvola import binarize as sauvola

METHODS = {
    "fixed": fixed,
    "otsu": otsu,
    "sauvola": sauvola,
    "niblack": niblack,
    "cv2_otsu": cv2_otsu,
    "cv2_adaptive_mean": cv2_adaptive_mean,
    "cv2_adaptive_gaussian": cv2_adaptive_gaussian,
}


def get_method(name: str):
    """Return the binarize function registered under `name`."""
    if name not in METHODS:
        raise KeyError("Unknown method %r. Available: %s" % (name, sorted(METHODS)))
    return METHODS[name]
