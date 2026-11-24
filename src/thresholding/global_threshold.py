"""Fixed global threshold (baseline).

Role #4: Global methods.
"""
import numpy as np


def binarize(gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Pixels darker than `threshold` become 0 (text), the rest 255 (background). From scratch with NumPy."""
    raise NotImplementedError("global_threshold.binarize")
