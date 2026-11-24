"""Preprocessing steps applied before thresholding.

Role #2: Preprocessing.
All functions take and return 2D uint8 grayscale arrays.
"""
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a colour image (H x W x 3, BGR) to 2D uint8 grayscale. Pass through if already 2D."""
    raise NotImplementedError("preprocessing.to_gray")


def denoise(gray: np.ndarray, **params) -> np.ndarray:
    """Reduce noise (e.g. median or Gaussian filter) and return a grayscale image."""
    raise NotImplementedError("preprocessing.denoise")


def normalize_background(gray: np.ndarray, **params) -> np.ndarray:
    """Estimate and divide out uneven background (stains, shadows). Returns grayscale."""
    raise NotImplementedError("preprocessing.normalize_background")
