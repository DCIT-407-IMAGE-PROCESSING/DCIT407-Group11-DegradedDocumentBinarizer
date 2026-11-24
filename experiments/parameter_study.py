"""Study how Sauvola's window size and k affect the results.

Role #5: Sauvola and parameter study.

Usage (from the repository root):
    python experiments/parameter_study.py

Steps to implement:
1. Choose lists of window sizes (e.g. 11, 25, 51, 101) and k values (e.g. 0.1 to 0.5).
2. Run Sauvola for every combination on the test images.
3. Save the scores to results/tables/ and a heatmap or line chart to results/figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable


def main() -> None:
    raise NotImplementedError("parameter_study.main")


if __name__ == "__main__":
    main()
