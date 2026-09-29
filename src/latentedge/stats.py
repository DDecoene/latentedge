"""Inference for correlations and means of overlapping-label series.

A label spans `horizon` one-minute bars, so neighbouring labels share most of
their price path and the effective sample is about n / horizon. The naive
standard error 1/sqrt(n) is then far too small. These tools resample or
shift in blocks of the label length, which keeps that dependence intact:
a moving-block bootstrap for confidence intervals and a circular-shift
permutation test for "is this correlation distinguishable from zero".
"""

import numpy as np

NAN_INTERVAL = (float("nan"), float("nan"), float("nan"))


def effective_sample_size(n: int, horizon_bars: int) -> float:
    """Roughly how many independent labels n overlapping ones amount to."""
    return max(n / max(horizon_bars, 1), 1.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    denominator = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    return float(np.dot(a_c, b_c) / denominator) if denominator > 0 else float("nan")


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of one moving-block resample of length n (blocks wrap around)."""
    block_length = max(1, min(block_length, n))
    blocks = -(-n // block_length)
    starts = rng.integers(0, n, size=blocks)
    return ((starts[:, None] + np.arange(block_length)) % n).ravel()[:n]


def _interval(estimate: float, draws: np.ndarray, level: float) -> tuple[float, float, float]:
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(estimate) or draws.size == 0:
        return NAN_INTERVAL
    tail = (1.0 - level) / 2.0
    return estimate, float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def block_bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, block_length: int, n_boot: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float, float]:
    """(correlation, lower, upper): the pairs are resampled in blocks so each
    keeps its neighbours, and the percentile interval of the resampled
    correlations is returned. NaN throughout for a constant series."""
    estimate = _correlation(x, y)
    if not np.isfinite(estimate):
        return NAN_INTERVAL
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(len(x), block_length, rng)
        draws[i] = _correlation(x[idx], y[idx])
    return _interval(estimate, draws, level)


def block_bootstrap_mean(
    values: np.ndarray, block_length: int, n_boot: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float, float]:
    """(mean, lower, upper) with the same moving-block resampling."""
    if len(values) == 0:
        return NAN_INTERVAL
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = values[_block_indices(len(values), block_length, rng)].mean()
    return _interval(float(values.mean()), draws, level)


def permutation_pvalue(
    prediction: np.ndarray, outcome: np.ndarray, block_length: int, n_perm: int, rng: np.random.Generator
) -> float:
    """Two-sided p-value for "the correlation is zero". The null is built by
    circularly shifting the predictions against the outcomes by a random
    offset of at least one block: each series keeps its own autocorrelation
    and only the alignment between them is destroyed. The +1 in numerator
    and denominator keeps p away from exactly zero."""
    observed = _correlation(prediction, outcome)
    n = len(prediction)
    if not np.isfinite(observed) or n < 2 * block_length + 1:
        return float("nan")
    hits = 0
    for _ in range(n_perm):
        shift = int(rng.integers(block_length, n - block_length + 1))
        if abs(_correlation(np.roll(prediction, shift), outcome)) >= abs(observed):
            hits += 1
    return (1 + hits) / (1 + n_perm)
