"""Otsu's global thresholding, implemented from scratch.

Role #4: Global methods.

Idea: try every threshold t in 0..255 and pick the one that maximizes the
between-class variance  w0(t) * w1(t) * (mu0(t) - mu1(t))^2.
"""
import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    """Return the Otsu threshold (0-255). Build the histogram with NumPy, no cv2 calls."""
    raise NotImplementedError("otsu.otsu_threshold")


def binarize(gray: np.ndarray, **params) -> np.ndarray:
    """Binarize using the Otsu threshold. Returns 0 = text, 255 = background."""
    raise NotImplementedError("otsu.binarize")
