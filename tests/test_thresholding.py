"""Sanity checks: our from-scratch methods vs OpenCV.

Role #7: OpenCV baselines and tests.

Tests are skipped automatically while a function is still a stub, so the suite
stays green during development. Once a module is implemented, its test runs.
Run from the repository root:  pytest tests/
"""
import importlib

import cv2
import numpy as np
import pytest

from src.thresholding import METHODS, get_method
from src.thresholding import opencv_baselines

# The package re-exports each `binarize` under the module's own name (e.g. `sauvola`),
# so load the modules themselves this way to reach their helper functions.
otsu_module = importlib.import_module("src.thresholding.otsu")
sauvola_module = importlib.import_module("src.thresholding.sauvola")

ALL_METHODS = sorted(METHODS)
OPENCV_METHODS = ["cv2_otsu", "cv2_adaptive_mean", "cv2_adaptive_gaussian"]


# --------------------------------------------------------------------------- helpers

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


def call_or_skip(func, *args, **kwargs):
    """Call any function, or skip the test if it is still a stub."""
    try:
        return func(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("%s not implemented yet" % func.__qualname__)


def opencv_local_mean_std(gray: np.ndarray, window: int):
    """Reference local mean and standard deviation over a window x window box, using cv2.boxFilter."""
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(g * g, -1, (window, window), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return mean, std


def interior(array: np.ndarray, window: int) -> np.ndarray:
    """Drop a border of window // 2 pixels, where implementations may pad the image differently."""
    r = window // 2
    return array[r:-r, r:-r]


# --------------------------------------------------------------------------- contract (every method)

@pytest.mark.parametrize("name", ALL_METHODS)
def test_output_is_binary_with_same_shape(name):
    gray = synthetic_page()
    out = run_or_skip(name, gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


@pytest.mark.parametrize("name", ALL_METHODS)
def test_input_is_not_modified(name):
    gray = synthetic_page()
    original = gray.copy()
    run_or_skip(name, gray)
    np.testing.assert_array_equal(gray, original)


@pytest.mark.parametrize("name", ALL_METHODS)
def test_dark_strokes_become_text(name):
    """Polarity check: dark ink must come out as 0 (text), never the other way round."""
    gray = synthetic_page()
    out = run_or_skip(name, gray)
    assert out[31, 50] == 0       # inside the horizontal stroke
    assert out[80, 51] == 0       # inside the vertical stroke


# Niblack is left out: it is known to mark noise in plain background as text.
@pytest.mark.parametrize("name", [m for m in ALL_METHODS if m != "niblack"])
def test_plain_background_stays_white(name):
    gray = synthetic_page()
    out = run_or_skip(name, gray)
    assert out[10, 10] == 255
    assert out[120, 120] == 255


@pytest.mark.parametrize("name", ["fixed", "sauvola", "niblack", "cv2_adaptive_mean", "cv2_adaptive_gaussian"])
def test_blank_page_has_no_text(name):
    """A uniform page has no text, so it should come out all white (and not crash on zero variance)."""
    gray = np.full((64, 64), 200, dtype=np.uint8)
    out = run_or_skip(name, gray)
    assert np.all(out == 255)


# --------------------------------------------------------------------------- from scratch vs OpenCV

def test_otsu_threshold_matches_opencv():
    """Our Otsu threshold value should be within 1 grey level of OpenCV's."""
    gray = synthetic_page()
    ours = call_or_skip(otsu_module.otsu_threshold, gray)
    ref = opencv_baselines.otsu_threshold(gray)
    assert abs(int(ours) - ref) <= 1, "our threshold %s, OpenCV %s" % (ours, ref)


def test_otsu_matches_opencv():
    """Our Otsu output should be (almost) identical to cv2.threshold with THRESH_OTSU."""
    gray = synthetic_page()
    ours = run_or_skip("otsu", gray)
    ref = run_or_skip("cv2_otsu", gray)
    differing = np.mean(ours != ref)
    assert differing < 0.01, "more than 1%% of pixels differ from OpenCV (%.3f)" % differing


def test_fixed_threshold_matches_opencv():
    """A fixed threshold of t should agree with cv2.threshold(gray, t - 1, ...) (pixels < t are text)."""
    gray = synthetic_page()
    for t in (100, 128, 200):
        ours = run_or_skip("fixed", gray, threshold=t)
        _, ref = cv2.threshold(gray, t - 1, 255, cv2.THRESH_BINARY)
        np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("window", [15, 25])
def test_sauvola_local_mean_std_matches_opencv(window):
    """The integral-image mean/std should equal a cv2.boxFilter reference away from the borders."""
    gray = synthetic_page()
    mean, std = call_or_skip(sauvola_module.local_mean_std, gray, window)
    ref_mean, ref_std = opencv_local_mean_std(gray, window)
    assert mean.shape == gray.shape and std.shape == gray.shape
    np.testing.assert_allclose(interior(mean, window), interior(ref_mean, window), atol=1e-3)
    np.testing.assert_allclose(interior(std, window), interior(ref_std, window), atol=1e-3)


def test_sauvola_matches_opencv_reference():
    """Sauvola built on our code should match the same formula built on cv2.boxFilter."""
    window, k, R = 25, 0.2, 128.0
    gray = synthetic_page()
    ours = run_or_skip("sauvola", gray, window=window, k=k, R=R)
    mean, std = opencv_local_mean_std(gray, window)
    ref = np.where(gray < mean * (1 + k * (std / R - 1)), 0, 255).astype(np.uint8)
    differing = np.mean(interior(ours, window) != interior(ref, window))
    assert differing < 0.01, "more than 1%% of pixels differ from the reference (%.3f)" % differing


def test_niblack_matches_opencv_reference():
    """Niblack built on our code should match the same formula built on cv2.boxFilter."""
    window, k = 25, -0.2
    gray = synthetic_page()
    ours = run_or_skip("niblack", gray, window=window, k=k)
    mean, std = opencv_local_mean_std(gray, window)
    ref = np.where(gray < mean + k * std, 0, 255).astype(np.uint8)
    differing = np.mean(interior(ours, window) != interior(ref, window))
    assert differing < 0.01, "more than 1%% of pixels differ from the reference (%.3f)" % differing


def test_sauvola_finds_the_strokes():
    """The dark strokes should be classified as text (0) and most of the page as background (255)."""
    gray = synthetic_page()
    out = run_or_skip("sauvola", gray)
    assert out[31, 50] == 0       # inside the horizontal stroke
    assert out[10, 10] == 255     # plain background


def test_local_methods_beat_global_on_uneven_lighting():
    """On a page with a strong shadow, Sauvola should keep text that a single global threshold loses."""
    gray = synthetic_page().astype(np.float64)
    gray[:, :64] *= 0.35                           # dark shadow over the left half
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    text = np.zeros(gray.shape, dtype=bool)
    text[30:34, 10:110] = True
    text[60:100, 50:54] = True

    def errors(binary):
        return int(np.sum((binary == 0) != text))

    local = run_or_skip("sauvola", gray)
    global_ = run_or_skip("cv2_otsu", gray)
    assert errors(local) < errors(global_)


# --------------------------------------------------------------------------- OpenCV baselines themselves

def test_cv2_otsu_threshold_splits_two_levels():
    """On a two-level image, Otsu's threshold must lie between the two levels."""
    gray = np.full((50, 50), 200, dtype=np.uint8)
    gray[:, :20] = 50
    t = opencv_baselines.otsu_threshold(gray)
    assert 50 <= t < 200
    out = opencv_baselines.binarize_otsu(gray)
    assert np.all(out[:, :20] == 0) and np.all(out[:, 20:] == 255)


@pytest.mark.parametrize("name", OPENCV_METHODS)
def test_opencv_methods_ignore_unknown_params(name):
    """Every method accepts **params, so the pipeline can pass one shared params dict."""
    gray = synthetic_page()
    out = get_method(name)(gray, some_unused_param=1)
    assert out.shape == gray.shape


@pytest.mark.parametrize("name", OPENCV_METHODS)
def test_opencv_methods_reject_colour_images(name):
    colour = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        get_method(name)(colour)


@pytest.mark.parametrize("name", OPENCV_METHODS)
def test_opencv_methods_reject_non_uint8(name):
    with pytest.raises(ValueError):
        get_method(name)(synthetic_page().astype(np.float64))


@pytest.mark.parametrize("name", ["cv2_adaptive_mean", "cv2_adaptive_gaussian"])
@pytest.mark.parametrize("block_size", [0, 1, 4, 24, 25.5])
def test_adaptive_rejects_bad_block_size(name, block_size):
    with pytest.raises(ValueError):
        get_method(name)(synthetic_page(), block_size=block_size)


@pytest.mark.parametrize("name", ["cv2_adaptive_mean", "cv2_adaptive_gaussian"])
def test_larger_C_marks_fewer_text_pixels(name):
    """C is subtracted from the local mean, so a larger C means a stricter test for text."""
    gray = synthetic_page()
    loose = get_method(name)(gray, C=2)
    strict = get_method(name)(gray, C=20)
    assert np.sum(strict == 0) <= np.sum(loose == 0)
