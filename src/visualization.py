"""Figures for the report and demo.

Role #10: Visualization and demo.
"""
from typing import Dict, List

import numpy as np


def side_by_side(original: np.ndarray, results: Dict[str, np.ndarray], ground_truth=None, save_path=None):
    """Show the original, each method's output (and optionally the ground truth) side by side."""
    raise NotImplementedError("visualization.side_by_side")


def comparison_chart(table, metric: str = "f_measure", save_path=None):
    """Bar chart comparing methods on one metric, from a results table (e.g. a pandas DataFrame)."""
    raise NotImplementedError("visualization.comparison_chart")
