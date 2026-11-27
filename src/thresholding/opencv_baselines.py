"""OpenCV reference implementations, used to verify our own code and as extra comparison methods.

Role #7: OpenCV baselines and tests.
Same contract as every method: binarize(gray, **params) -> binary (0 = text, 255 = background).

OpenCV's THRESH_BINARY sets a pixel to 255 when it is ABOVE the threshold and
to 0 otherwise, which already matches our convention (dark text -> 0).
"""
import cv2
import numpy as np


def _check_gray(gray: np.ndarray) -> np.ndarray:
    """Make sure `gray` is a 2D uint8 image (the project-wide input contract) and return it."""
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError("expected a 2D grayscale image, got shape %s" % (gray.shape,))
    if gray.dtype != np.uint8:
        raise ValueError("expected a uint8 image, got dtype %s" % gray.dtype)
    return gray


def _check_block_size(block_size: int) -> int:
    """cv2.adaptiveThreshold needs an odd window size of at least 3 pixels."""
    if int(block_size) != block_size or block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3, got %r" % (block_size,))
    return int(block_size)


def otsu_threshold(gray: np.ndarray) -> int:
    """Return the threshold value OpenCV's Otsu picks (0-255).

    Useful in the tests to compare against our own `otsu.otsu_threshold`.
    """
    gray = _check_gray(gray)
    # With THRESH_OTSU the `thresh` argument (0) is ignored and the chosen value is returned.
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(threshold)


def binarize_otsu(gray: np.ndarray, **params) -> np.ndarray:
    """Otsu thresholding via cv2.threshold (cv2.THRESH_OTSU).

    Pixels at or below the Otsu threshold become 0 (text), the rest 255 (background).
    Extra `params` are accepted and ignored, so every method can be called the same way.
    """
    gray = _check_gray(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive_mean(gray: np.ndarray, block_size: int = 25, C: float = 10, **params) -> np.ndarray:
    """cv2.adaptiveThreshold with ADAPTIVE_THRESH_MEAN_C.

    Each pixel's threshold is the mean of its `block_size` x `block_size` neighbourhood minus `C`.
    Pixels at or below that threshold become 0 (text), the rest 255 (background).
    """
    gray = _check_gray(gray)
    block_size = _check_block_size(block_size)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)


def binarize_adaptive_gaussian(gray: np.ndarray, block_size: int = 25, C: float = 10, **params) -> np.ndarray:
    """cv2.adaptiveThreshold with ADAPTIVE_THRESH_GAUSSIAN_C.

    Same as the mean version, but the neighbourhood mean is Gaussian-weighted,
    so pixels near the centre count more.
    """
    gray = _check_gray(gray)
    block_size = _check_block_size(block_size)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
