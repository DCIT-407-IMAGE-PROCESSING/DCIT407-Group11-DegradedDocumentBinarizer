"""Run every method on every image and save a results table.

Role #9: Pipeline and experiments.

Usage (from the repository root):
    python experiments/run_comparison.py

Steps to implement:
1. Find all (image, ground truth) pairs with src.io_utils.list_image_pairs.
2. For each pair, call src.pipeline.run_on_image with the methods to compare.
3. Collect the results in a pandas DataFrame and save it to results/tables/comparison.csv.
4. Print a short summary (mean F-measure and PSNR per method).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable


def main() -> None:
    raise NotImplementedError("run_comparison.main")


if __name__ == "__main__":
    main()
