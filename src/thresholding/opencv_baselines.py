"""OpenCV reference implementations, used to verify our own code and as extra comparison methods.

Role #7: OpenCV baselines and tests.
Same contract as every method: binarize(gray, **params) -> binary (0 = text, 255 = background).
"""
import numpy as np


def binarize_otsu(gray: np.ndarray, **params) -> np.ndarray:
    """Otsu thresholding via cv2.threshold (cv2.THRESH_OTSU)."""
    raise NotImplementedError("opencv_baselines.binarize_otsu")


def binarize_adaptive_mean(gray: np.ndarray, block_size: int = 25, C: float = 10, **params) -> np.ndarray:
    """cv2.adaptiveThreshold with ADAPTIVE_THRESH_MEAN_C."""
    raise NotImplementedError("opencv_baselines.binarize_adaptive_mean")


def binarize_adaptive_gaussian(gray: np.ndarray, block_size: int = 25, C: float = 10, **params) -> np.ndarray:
    """cv2.adaptiveThreshold with ADAPTIVE_THRESH_GAUSSIAN_C."""
    raise NotImplementedError("opencv_baselines.binarize_adaptive_gaussian")
