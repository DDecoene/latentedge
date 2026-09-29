import numpy as np
import pytest

from latentedge.stats import (
    block_bootstrap_correlation,
    block_bootstrap_mean,
    effective_sample_size,
    permutation_pvalue,
)


def _overlapping_noise(n: int, window: int, rng: np.random.Generator) -> np.ndarray:
    """A series with the autocorrelation of overlapping labels: each value is
    the sum of the last `window` independent shocks."""
    shocks = rng.standard_normal(n + window)
    return np.convolve(shocks, np.ones(window), mode="valid")[:n]


def test_effective_sample_size_divides_by_the_overlap():
    assert effective_sample_size(43_000, 30) == pytest.approx(43_000 / 30)
    assert effective_sample_size(10, 30) == 1.0


def test_bootstrap_interval_contains_a_real_correlation():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(5_000)
    outcome = 0.5 * signal + rng.standard_normal(5_000)
    est, lo, hi = block_bootstrap_correlation(signal, outcome, block_length=10, n_boot=300, rng=rng)
    assert lo < est < hi
    assert lo > 0.3 and hi < 0.6


def test_bootstrap_interval_is_wide_for_overlapping_labels():
    """With overlapping labels the interval must be much wider than the naive
    1/sqrt(n) error, which is the reason to use a block bootstrap at all."""
    rng = np.random.default_rng(1)
    n, window = 20_000, 30
    prediction = _overlapping_noise(n, window, rng)
    outcome = _overlapping_noise(n, window, rng)
    _, lo, hi = block_bootstrap_correlation(prediction, outcome, block_length=window, n_boot=300, rng=rng)
    naive_half_width = 1.96 / np.sqrt(n)
    assert (hi - lo) / 2 > 2 * naive_half_width


def test_permutation_rejects_a_planted_signal():
    rng = np.random.default_rng(2)
    n, window = 8_000, 30
    hidden = _overlapping_noise(n, window, rng)
    prediction = hidden + rng.standard_normal(n) * 3
    outcome = hidden
    assert permutation_pvalue(prediction, outcome, block_length=window, n_perm=400, rng=rng) < 0.01


def test_permutation_does_not_reject_independent_series():
    """Independent overlapping series must not look significant more than a
    fraction of the time; a naive test would flag most of them."""
    rejections = 0
    trials = 40
    for seed in range(trials):
        rng = np.random.default_rng(100 + seed)
        prediction = _overlapping_noise(4_000, 30, rng)
        outcome = _overlapping_noise(4_000, 30, rng)
        if permutation_pvalue(prediction, outcome, block_length=30, n_perm=200, rng=rng) < 0.05:
            rejections += 1
    assert rejections <= 0.2 * trials


def test_permutation_pvalue_is_never_zero():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1_000)
    p = permutation_pvalue(x, x, block_length=5, n_perm=99, rng=rng)
    assert p == pytest.approx(1 / 100)


def test_bootstrap_mean_interval_covers_the_true_mean():
    rng = np.random.default_rng(4)
    values = rng.standard_normal(6_000) + 0.2
    est, lo, hi = block_bootstrap_mean(values, block_length=5, n_boot=300, rng=rng)
    assert lo < est < hi
    assert lo < 0.2 < hi


def test_degenerate_input_gives_nan_not_an_error():
    rng = np.random.default_rng(5)
    flat = np.zeros(200)
    est, lo, hi = block_bootstrap_correlation(flat, np.arange(200.0), block_length=5, n_boot=20, rng=rng)
    assert np.isnan(est) and np.isnan(lo) and np.isnan(hi)
    assert np.isnan(permutation_pvalue(flat, np.arange(200.0), block_length=5, n_perm=20, rng=rng))
