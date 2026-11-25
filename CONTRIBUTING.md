# Contributing (Group 11)

## Workflow

1. Pull the latest `main` before you start: `git pull`
2. Create your own branch: `git checkout -b <module>-<yourname>` (e.g. `otsu-khojoe`)
3. Work only in the files you own (see the table below).
4. Commit often, from **your own GitHub account** (commit history is used to confirm individual contribution).
5. Push your branch and open a Pull Request into `main`. At least one other member reviews it.
6. `main` must always run. Do not push broken code to it.

## Shared conventions (do not change without telling the group)

- Grayscale images: 2D `uint8` arrays (0-255).
- Binary images: 2D `uint8` arrays with only two values: **0 = text (black), 255 = background (white)**. Ground-truth images use the same convention.
- Every thresholding method exposes `binarize(gray, **params) -> binary`.
- Comment your code and write a docstring for every function.
- Implement the core algorithms from scratch with NumPy. OpenCV is only used for the reference comparison.

## Roles

| # | Area | Files | Assigned to |
|---|---|---|---|
| 1 | Data and I/O | `src/io_utils.py`, `data/README.md` | |
| 2 | Preprocessing | `src/preprocessing.py` | |
| 3 | Postprocessing | `src/postprocessing.py` | |
| 4 | Global methods | `src/thresholding/global_threshold.py`, `src/thresholding/otsu.py` | |
| 5 | Sauvola and parameter study | `src/thresholding/sauvola.py`, `experiments/parameter_study.py` | |
| 6 | Niblack and other local methods | `src/thresholding/niblack.py` | |
| 7 | OpenCV baselines and tests | `src/thresholding/opencv_baselines.py`, `tests/test_thresholding.py` | |
| 8 | Metrics | `src/metrics.py` | Anneta D. Dzibolosu (@annetadd) |
| 9 | Pipeline and experiments | `src/pipeline.py`, `experiments/run_comparison.py` | |
| 10 | Visualization and demo | `src/visualization.py`, `notebooks/demo.ipynb` | |
| 11 | Report and slides | `report/`, `slides/` | |

Everyone also writes the theory section for their own method in the report.
