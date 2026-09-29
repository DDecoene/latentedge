"""The robustness study: does the direction signal survive when the evidence is
gathered more than once?

The main pipeline trains one model on one chronological split and judges it on
one 30-day test window. That leaves three questions open: is the correlation
distinguishable from zero given overlapping labels, does it depend on the
seed or the window, and does a longer horizon change anything. This module
answers them on labeled bars by

- walk-forward evaluation: several test folds in a row, each judged by a model
  trained only on the bars before it, with a purge of one label length
  between training, validation and test so overlapping labels cannot leak;
- several seeds per fold, reported as spread and as an ensemble;
- block-bootstrap intervals and a permutation test on the pooled
  out-of-sample predictions (see stats.py);
- the net return of the model's top slices against trading every bar, with
  slice cutoffs fixed on each fold's own validation window.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
import pandas as pd

from latentedge import config
from latentedge.features import compute_feature_stats, standardize_features, standardize_value, unstandardize_value
from latentedge.model import NetReturnRegressor, train as train_model
from latentedge.stats import block_bootstrap_correlation, block_bootstrap_mean, effective_sample_size, permutation_pvalue

BASE_HORIZON_MINUTES = 30
BASE_BARRIER_STDS = 2.0
MIN_TRAIN_ROWS = 500
MIN_VALIDATE_ROWS = 100
MIN_TEST_ROWS = 100


def horizon_barrier_stds(horizon_minutes: float) -> float:
    """The take-profit/stop-loss band for a horizon, in standard deviations
    of the one-minute return. A random walk's typical move grows with the
    square root of time, so a band held at 2 std would be hit within minutes
    and a longer horizon would change nothing; the band is scaled from the
    30-minute default at 2 std."""
    return BASE_BARRIER_STDS * float(np.sqrt(horizon_minutes / BASE_HORIZON_MINUTES))


@dataclass(frozen=True)
class StudyConfig:
    horizons_minutes: list[int]
    seeds: list[int]
    folds: int
    test_share: float
    validate_fraction: float
    epochs: int
    hidden: tuple[int, ...]
    n_boot: int
    n_perm: int
    top_fractions: list[float]
    block_seed: int = 0
    learning_rate: float = 0.001


class Fold(NamedTuple):
    train: slice
    validate: slice
    test: slice


def walk_forward_folds(
    n_rows: int, folds: int, test_share: float, validate_fraction: float, purge_bars: int
) -> list[Fold]:
    """Expanding-window folds. The last `test_share` of the rows is cut into
    `folds` consecutive test blocks. Each fold trains on everything before its
    test block, less a validation stretch (the trailing `validate_fraction`)
    used for early stopping, with `purge_bars` rows dropped between train and
    validate and between validate and test: a label reaches `purge_bars` ahead,
    so without the gap the last training labels would overlap the first
    validation or test bars."""
    test_start = int(n_rows * (1.0 - test_share))
    width = (n_rows - test_start) // folds
    result: list[Fold] = []
    for k in range(folds):
        start = test_start + k * width
        stop = n_rows if k == folds - 1 else start + width
        before_test = start - purge_bars
        validate_len = int(before_test * validate_fraction)
        validate_start = before_test - validate_len
        train_stop = validate_start - purge_bars
        if train_stop < MIN_TRAIN_ROWS or validate_len < MIN_VALIDATE_ROWS or stop - start < MIN_TEST_ROWS:
            raise ValueError(
                f"too few rows ({n_rows}) for {folds} walk-forward folds: fold {k + 1} would have "
                f"{max(train_stop, 0)} train, {validate_len} validate and {stop - start} test rows."
            )
        result.append(Fold(slice(0, train_stop), slice(validate_start, before_test), slice(start, stop)))
    return result


@dataclass
class StudyObserver:
    """What a caller can watch: stage messages (label, done, total), each
    finished (horizon, seed, fold) row, each finished horizon summary. Any of
    them may raise to abort."""

    on_stage: Callable[..., None] = lambda *_: None
    on_row: Callable[[dict[str, Any]], None] = lambda _row: None
    on_summary: Callable[[dict[str, Any]], None] = lambda _summary: None

    def __call__(self, *args: Any) -> None:
        self.on_stage(*args)


@dataclass
class _FoldRun:
    test_predictions: np.ndarray
    validate_predictions: np.ndarray


def _fit_and_predict(
    bars: pd.DataFrame, feature_columns: list[str], fold: Fold, seed: int, cfg: StudyConfig
) -> _FoldRun:
    """One model, trained on the fold's train rows to predict the gross return
    (standardized with train-only statistics), stopped on its validate rows;
    returns its predictions, in raw return units, for validate and test."""
    train_rows, validate_rows, test_rows = bars.iloc[fold.train], bars.iloc[fold.validate], bars.iloc[fold.test]
    stats = compute_feature_stats(train_rows, [*feature_columns, "gross_return"])

    def x_of(rows: pd.DataFrame) -> np.ndarray:
        return standardize_features(rows, feature_columns, stats)[feature_columns].to_numpy(dtype="float32")

    def y_of(rows: pd.DataFrame) -> np.ndarray:
        return standardize_value(rows["gross_return"].to_numpy(dtype="float32"), stats["gross_return"])

    mx.random.seed(seed)
    model = NetReturnRegressor(input_dim=len(feature_columns), hidden=cfg.hidden)
    train_model(
        model, x_of(train_rows), y_of(train_rows), epochs=cfg.epochs, learning_rate=cfg.learning_rate,
        validation=(x_of(validate_rows), y_of(validate_rows)),
    )

    def predict(rows: pd.DataFrame) -> np.ndarray:
        raw = np.array(model(mx.array(x_of(rows))))
        return unstandardize_value(raw, stats["gross_return"])

    return _FoldRun(test_predictions=predict(test_rows), validate_predictions=predict(validate_rows))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return float("nan")


def _interval(estimate: tuple[float, float, float]) -> dict[str, float]:
    return {"mean": estimate[0], "ci_low": estimate[1], "ci_high": estimate[2]}


def run_horizon(
    bars: pd.DataFrame, feature_columns: list[str], horizon_minutes: int, cfg: StudyConfig, observer: StudyObserver,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every (seed, fold) at one horizon, then the pooled read of them.
    `bars` are labeled bars with the feature columns, gross_return and
    net_return. Returns the per-(seed, fold) rows and the horizon summary."""
    bars = bars.sort_values("bar_start", kind="stable").reset_index(drop=True)
    block = max(1, round(horizon_minutes * 60 / config.BAR_INTERVAL_SECONDS))
    folds = walk_forward_folds(len(bars), cfg.folds, cfg.test_share, cfg.validate_fraction, purge_bars=block)
    rng = np.random.default_rng(cfg.block_seed)

    runs: dict[tuple[int, int], _FoldRun] = {}
    rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(folds)
    for seed in cfg.seeds:
        for k, fold in enumerate(folds):
            observer(f"{horizon_minutes} min: seed {seed}, fold {k + 1}/{len(folds)}", len(rows), total)
            run = _fit_and_predict(bars, feature_columns, fold, seed, cfg)
            runs[(seed, k)] = run
            test_rows = bars.iloc[fold.test]
            gross = test_rows["gross_return"].to_numpy(dtype="float64")
            cost = gross - test_rows["net_return"].to_numpy(dtype="float64")
            row = {
                "horizon_minutes": horizon_minutes, "seed": seed, "fold": k + 1, "n": len(test_rows),
                "test_start": int(test_rows["bar_start"].iloc[0]), "test_end": int(test_rows["bar_start"].iloc[-1]),
                "gross_correlation": _correlation(run.test_predictions, gross),
                "cost_correlation": _correlation(run.test_predictions, cost),
            }
            rows.append(row)
            observer.on_row(row)

    def ensemble(pick: Callable[[_FoldRun], np.ndarray], k: int) -> np.ndarray:
        return np.mean([pick(runs[(seed, k)]) for seed in cfg.seeds], axis=0)

    test_gross = np.concatenate([bars.iloc[f.test]["gross_return"].to_numpy(dtype="float64") for f in folds])
    test_net = np.concatenate([bars.iloc[f.test]["net_return"].to_numpy(dtype="float64") for f in folds])
    test_cost = test_gross - test_net
    pooled_predictions = np.concatenate([ensemble(lambda r: r.test_predictions, k) for k in range(len(folds))])

    estimate, low, high = block_bootstrap_correlation(pooled_predictions, test_gross, block, cfg.n_boot, rng)
    pooled = {
        "n": len(test_gross), "effective_n": effective_sample_size(len(test_gross), block),
        "gross_correlation": estimate, "ci_low": low, "ci_high": high,
        "p_value": permutation_pvalue(pooled_predictions, test_gross, block, cfg.n_perm, rng),
        "cost_correlation": _correlation(pooled_predictions, test_cost),
    }

    per_seed = [
        _correlation(np.concatenate([runs[(seed, k)].test_predictions for k in range(len(folds))]), test_gross)
        for seed in cfg.seeds
    ]
    seed_correlations = {
        "values": per_seed, "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"),
        "min": float(np.min(per_seed)), "max": float(np.max(per_seed)),
    }
    fold_correlations = [
        _correlation(ensemble(lambda r: r.test_predictions, k), bars.iloc[folds[k].test]["gross_return"].to_numpy())
        for k in range(len(folds))
    ]

    # Top slices: the cutoff for each fold is the quantile of that fold's own
    # validation predictions, so it is fixed before the fold's test bars.
    top_slices: list[dict[str, Any]] = []
    for fraction in cfg.top_fractions:
        selected = np.concatenate([
            ensemble(lambda r: r.test_predictions, k)
            > np.quantile(ensemble(lambda r: r.validate_predictions, k), 1.0 - fraction)
            for k in range(len(folds))
        ])
        net_sel = test_net[selected]
        net_mean = _interval(block_bootstrap_mean(net_sel, block, cfg.n_boot, rng)) if net_sel.size else _interval(
            (float("nan"),) * 3
        )
        top_slices.append({
            "top_fraction": fraction, "n": int(selected.sum()),
            "mean_gross": float(test_gross[selected].mean()) if selected.any() else float("nan"),
            "mean_cost": float(test_cost[selected].mean()) if selected.any() else float("nan"),
            "mean_net": net_mean["mean"], "mean_net_ci_low": net_mean["ci_low"], "mean_net_ci_high": net_mean["ci_high"],
        })

    summary = {
        "horizon_minutes": horizon_minutes, "barrier_stds": horizon_barrier_stds(horizon_minutes),
        "bars": len(bars), "folds": len(folds), "seeds": list(cfg.seeds),
        "pooled": pooled, "seed_correlations": seed_correlations, "fold_correlations": fold_correlations,
        "top_slices": top_slices,
        "all_bars": {
            "n": len(test_gross), "mean_gross": float(test_gross.mean()), "mean_cost": float(test_cost.mean()),
            "mean_net": float(test_net.mean()),
        },
    }
    observer.on_summary(summary)
    return rows, summary


def write_study(
    out_dir: Path, meta: dict[str, Any], rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
    git_state: str | None, now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    """Saves one study as <out_dir>/<UTC timestamp>.json (metadata, every
    (horizon, seed, fold) row and each horizon's summary), and a .csv of the
    rows. Never overwrites an earlier study."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now().strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}.json"
    counter = 1
    while path.exists():
        counter += 1
        path = out_dir / f"{stamp}-{counter}.json"
    document = {"created_at": now().isoformat(), "git": git_state, **meta, "rows": rows, "summaries": summaries}
    path.write_text(json.dumps(document, indent=2))
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    return path


def describe_summary(summary: dict[str, Any]) -> str:
    """One line per horizon: the pooled out-of-sample correlation with its
    interval and permutation p-value, and how much the seeds disagree."""
    pooled, seeds = summary["pooled"], summary["seed_correlations"]
    return (
        f"{summary['horizon_minutes']} min: gross corr {pooled['gross_correlation']:+.3f} "
        f"[{pooled['ci_low']:+.3f}, {pooled['ci_high']:+.3f}], p={pooled['p_value']:.3f}, "
        f"seeds {seeds['min']:+.3f}..{seeds['max']:+.3f}, effective n {pooled['effective_n']:,.0f}"
    )


def describe_study(result: dict[str, Any]) -> str:
    lines = [describe_summary(s) for s in result["summaries"]]
    return f"{len(result['rows'])} runs saved to {result['path']}\n" + "\n".join(lines)
