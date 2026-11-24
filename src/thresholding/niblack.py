"""Niblack local thresholding, implemented from scratch (optional comparison method).

Role #6: Niblack and other local methods.

Per-pixel threshold:  T(x, y) = m(x, y) + k * s(x, y)   (k is usually negative, e.g. -0.2)
"""
import numpy as np


def binarize(gray: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
    """Binarize using Niblack's per-pixel threshold. Returns 0 = text, 255 = background."""
    raise NotImplementedError("niblack.binarize")
