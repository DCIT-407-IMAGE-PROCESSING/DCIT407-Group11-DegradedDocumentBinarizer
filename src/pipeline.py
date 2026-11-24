"""Chains load -> preprocess -> binarize -> postprocess -> evaluate.

Role #9: Pipeline and experiments.
"""
from typing import Dict, List, Optional

import numpy as np

from src import metrics
from src.thresholding import get_method


def run_method(name: str, gray: np.ndarray, **params) -> np.ndarray:
    """Binarize `gray` with the registered method `name`."""
    return get_method(name)(gray, **params)


def evaluate(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Score one prediction against its ground truth. Returns a dict of metric values."""
    precision, recall = metrics.precision_recall(pred, gt)
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": metrics.f_measure(pred, gt),
        "psnr": metrics.psnr(pred, gt),
    }


def run_on_image(
    image_path,
    gt_path,
    method_names: List[str],
    params: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    """Run each method on one image, evaluate against the ground truth, time it.

    Returns one result dict per method: image, method, params, metrics, seconds.
    """
    raise NotImplementedError("pipeline.run_on_image")
