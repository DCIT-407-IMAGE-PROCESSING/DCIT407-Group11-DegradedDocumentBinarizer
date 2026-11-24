"""Evaluation metrics. Every method is scored by this same code.

Role #8: Metrics.

Inputs are binary uint8 arrays (0 = text, 255 = background). "Positive" means a TEXT pixel (value 0).
"""
import time
from typing import Callable, Tuple

import numpy as np


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Return (precision, recall) for text pixels. Precision = TP/(TP+FP), Recall = TP/(TP+FN)."""
    raise NotImplementedError("metrics.precision_recall")


def f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """F-measure = 2 * P * R / (P + R)."""
    raise NotImplementedError("metrics.f_measure")


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR = 10 * log10(255^2 / MSE). Handle MSE == 0 sensibly (identical images)."""
    raise NotImplementedError("metrics.psnr")


def time_method(func: Callable, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start
