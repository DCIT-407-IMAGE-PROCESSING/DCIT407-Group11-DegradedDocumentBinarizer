# Degraded Document Binarizer

**DCIT 407 Image Processing | Semester Project 2026/27 | Group 11**

A Python project that cleans old, stained and faded document images by separating text (black) from background (white), and compares several thresholding methods against ground-truth images.

The central algorithms (Otsu and Sauvola thresholding) are implemented **from scratch using NumPy**, then compared with their OpenCV equivalents.

## Table of Contents

1. [Overview](#overview)
2. [Methods](#methods)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Team](#team)
10. [References](#references)

## Overview

Binarization converts a grayscale image into a two-value image: black for text and white for background. A single global threshold works on clean scans but fails on degraded documents (stains, faded ink, uneven lighting, bleed-through). This project shows where global methods break and how local (adaptive) methods handle those cases, using measurable results.

Real-world uses include digitizing archives and historical records, and preparing scans for OCR.

## Methods

| Method                           | Type            | Implementation                                      |
| -------------------------------- | --------------- | --------------------------------------------------- |
| Fixed threshold                  | Global baseline | From scratch (NumPy)                                |
| Otsu                             | Global          | From scratch (NumPy), compared with `cv2.threshold` |
| Sauvola                          | Local           | From scratch (NumPy, integral images)               |
| Niblack (optional)               | Local           | From scratch (NumPy)                                |
| OpenCV adaptive (mean, Gaussian) | Local           | `cv2.adaptiveThreshold` (reference)                 |

**Otsu** chooses the threshold `t` that maximizes the between-class variance of the dark and light pixel groups.

**Sauvola** computes a threshold per pixel from the local mean `m` and local standard deviation `s`:

```
T(x, y) = m(x, y) * [ 1 + k * ( s(x, y) / R - 1 ) ]
```

with typical values `k` in [0.2, 0.5] and `R = 128` for 8-bit images. Local statistics are computed with integral images for speed.

## Project Structure

```
DCIT407-Group11-DegradedDocumentBinarizer/
├── README.md
├── CONTRIBUTING.md
├── conftest.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md              # dataset source, licence, download link
│   └── samples/               # a few images + ground truth only
├── src/
│   ├── io_utils.py            # load/save images, pair image with ground truth
│   ├── preprocessing.py       # grayscale, denoising, background normalization
│   ├── thresholding/
│   │   ├── global_threshold.py
│   │   ├── otsu.py
│   │   ├── niblack.py
│   │   ├── sauvola.py
│   │   └── opencv_baselines.py
│   ├── postprocessing.py      # small-blob removal, morphological cleanup
│   ├── metrics.py             # F-measure, precision, recall, PSNR, runtime
│   ├── pipeline.py            # load -> preprocess -> binarize -> evaluate
│   └── visualization.py       # side-by-side figures, comparison charts
├── experiments/
│   ├── run_comparison.py      # all methods on all images, saves a CSV
│   └── parameter_study.py     # window size / k sweeps for Sauvola
├── notebooks/
│   └── demo.ipynb             # live demo
├── tests/
│   └── test_thresholding.py   # from-scratch vs OpenCV sanity checks
├── results/
│   ├── figures/
│   └── tables/
├── report/
└── slides/
```

## Installation

Requires Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<organisation>/DCIT407-Group11-DegradedDocumentBinarizer.git
cd DCIT407-Group11-DegradedDocumentBinarizer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

> The commands below describe the intended workflow and will be kept up to date as the scripts are completed.

**Run the full comparison** (all methods on all images, writes a results table to `results/tables/`):

```bash
python experiments/run_comparison.py
```

**Run the Sauvola parameter study** (window size and `k` sweeps):

```bash
python experiments/parameter_study.py
```

**Run the tests** (checks the from-scratch implementations against OpenCV):

```bash
pytest tests/
```

**Live demo:** open `notebooks/demo.ipynb` in Jupyter or VS Code and run all cells.

## Dataset

We use the **DIBCO** (Document Image Binarization Contest) datasets: degraded document images paired with manually produced ground-truth binary images.

- Only a small set of sample images is included in `data/samples/`.
- The full dataset, its source, and its licence are documented in [`data/README.md`](data/README.md).
- Download the full dataset from the link given there and place it in the `data/` folder before running the full experiments.

## Evaluation

Every method is scored against the ground truth using the same code in `src/metrics.py`:

- **F-measure** (with precision and recall)
- **PSNR**
- **Processing time** per image

At least two methods and several parameter settings are compared in tables and charts, alongside side-by-side original and processed images.

## Results

_To be completed once experiments are finished (Week 11)._ This section will summarize the comparison table, key figures, and cases where each method fails.

## Team

| #   | Name                  | Student ID | GitHub  | Area                                   |
| --- | --------------------- | ---------- | ------- | -------------------------------------- |
| 1   |                       |            |         | Data and I/O                           |
| 2   |                       |            |         | Preprocessing                          |
| 3   |                       |            |         | Postprocessing                         |
| 4   |                       |            |         | Global methods (fixed threshold, Otsu) |
| 5   |                       |            |         | Sauvola and parameter study            |
| 6   |                       |            |         | Niblack and other local methods        |
| 7   |                       |            |         | OpenCV baselines and tests             |
| 8   | Anneta D. Dzibolosu   | 22130388   | @annetadd | Metrics                                |
| 9   | Immanuel Oheneba Debe | 22243130   | @Khojoe | Pipeline and experiments               |
| 10  |                       |            |         | Visualization and demo                 |
| 11  |                       |            |         | Report and slides                      |

## References

- DIBCO dataset pages and competition reports (add exact sources and years used).
- Otsu, N. (1979). A threshold selection method from gray-level histograms. _IEEE Transactions on Systems, Man, and Cybernetics_.
- Sauvola, J., and Pietikainen, M. (2000). Adaptive document image binarization. _Pattern Recognition_.
- Niblack, W. (1986). _An Introduction to Digital Image Processing_.
- OpenCV documentation: https://docs.opencv.org/
