"""Cleanup steps applied after thresholding.

Role #3: Postprocessing.
All functions take and return binary uint8 arrays (0 = text, 255 = background).
"""
import numpy as np


def remove_small_components(binary: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Remove isolated text-pixel blobs smaller than min_size pixels (noise)."""
    raise NotImplementedError("postprocessing.remove_small_components")


def morphological_cleanup(binary: np.ndarray, **params) -> np.ndarray:
    """Apply opening/closing to smooth strokes and remove speckle."""
    raise NotImplementedError("postprocessing.morphological_cleanup")
