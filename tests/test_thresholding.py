"""Sanity checks: our from-scratch methods vs OpenCV.

Role #7: OpenCV baselines and tests.

Tests are skipped automatically while a function is still a stub, so the suite
stays green during development. Once a module is implemented, its test runs.
Run from the repository root:  pytest tests/
"""
import numpy as np
import pytest

from src.thresholding import get_method


def synthetic_page(size: int = 128, seed: int = 0) -> np.ndarray:
    """A fake document: light background with a left-to-right brightness gradient and dark strokes."""
    rng = np.random.default_rng(seed)
    background = np.tile(np.linspace(180, 230, size), (size, 1))
    image = background + rng.normal(0, 3, (size, size))
    image[30:34, 10:110] = 40   # a horizontal "text" stroke
    image[60:100, 50:54] = 40   # a vertical "text" stroke
    return np.clip(image, 0, 255).astype(np.uint8)


def run_or_skip(name: str, gray: np.ndarray, **params) -> np.ndarray:
    """Run a method, or skip the test if it is not implemented yet."""
    try:
        return get_method(name)(gray, **params)
    except NotImplementedError:
        pytest.skip("%s not implemented yet" % name)


@pytest.mark.parametrize("name", ["fixed", "otsu", "sauvola", "niblack", "cv2_otsu",
                                  "cv2_adaptive_mean", "cv2_adaptive_gaussian"])
def test_output_is_binary_with_same_shape(name):
    gray = synthetic_page()
    out = run_or_skip(name, gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_otsu_matches_opencv():
    """Our Otsu output should be (almost) identical to cv2.threshold with THRESH_OTSU."""
    gray = synthetic_page()
    ours = run_or_skip("otsu", gray)
    ref = run_or_skip("cv2_otsu", gray)
    differing = np.mean(ours != ref)
    assert differing < 0.01, "more than 1%% of pixels differ from OpenCV (%.3f)" % differing


def test_sauvola_finds_the_strokes():
    """The dark strokes should be classified as text (0) and most of the page as background (255)."""
    gray = synthetic_page()
    out = run_or_skip("sauvola", gray)
    assert out[31, 50] == 0       # inside the horizontal stroke
    assert out[10, 10] == 255     # plain background
