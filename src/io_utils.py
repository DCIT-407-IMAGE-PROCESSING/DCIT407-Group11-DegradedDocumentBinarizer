"""Image input/output helpers.

Role #1: Data and I/O.

Project-wide conventions
------------------------
- Grayscale image: 2D uint8 array (values 0-255).
- Binary image: 2D uint8 array with only two values,
  0 = text (black) and 255 = background (white).
- Ground truth uses the same binary convention.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_gray(path) -> np.ndarray:
    """Load an image from disk and return it as a 2D uint8 grayscale array."""
    raise NotImplementedError("io_utils.load_gray")


def load_ground_truth(path) -> np.ndarray:
    """Load a ground-truth image and return a binary uint8 array (0 = text, 255 = background)."""
    raise NotImplementedError("io_utils.load_ground_truth")


def load_pair(image_path, gt_path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and its ground truth. Returns (gray, ground_truth)."""
    raise NotImplementedError("io_utils.load_pair")


def list_image_pairs(data_dir) -> List[Tuple[Path, Path]]:
    """Find all (image_path, ground_truth_path) pairs inside data_dir."""
    raise NotImplementedError("io_utils.list_image_pairs")


def save_image(path, image: np.ndarray) -> None:
    """Save an image array to disk, creating parent folders if needed."""
    raise NotImplementedError("io_utils.save_image")
