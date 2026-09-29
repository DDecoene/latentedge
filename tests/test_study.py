import numpy as np
import pandas as pd
import pytest

from latentedge.study import (
    StudyConfig,
    StudyObserver,
    horizon_barrier_stds,
    run_horizon,
    walk_forward_folds,
)

FEATURES = ["f1", "f2"]


def test_folds_are_ordered_disjoint_and_purged_from_what_they_test():
    folds = walk_forward_folds(n_rows=10_000, folds=4, test_share=0.4, validate_fraction=0.15, purge_bars=30)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train.stop + 30 <= fold.validate.start
        assert fold.validate.stop + 30 <= fold.test.start
        assert fold.train.start == 0
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test.stop == later.test.start
        assert later.train.stop > earlier.train.stop
    assert folds[0].test.start == 6_000
    assert folds[-1].test.stop == 10_000


def test_folds_refuse_data_too_short_to_train_on():
    with pytest.raises(ValueError, match="too few"):
        walk_forward_folds(n_rows=300, folds=5, test_share=0.5, validate_fraction=0.15, purge_bars=30)


def test_barrier_band_widens_with_the_square_root_of_the_horizon():
    assert horizon_barrier_stds(30) == pytest.approx(2.0)
    assert horizon_barrier_stds(120) == pytest.approx(4.0)
    assert horizon_barrier_stds(240) == pytest.approx(2.0 * np.sqrt(8))


def _labeled_bars(n: int, planted: float, seed: int) -> pd.DataFrame:
    """Bars whose gross return follows f1 with strength `planted`, plus noise
    that is autocorrelated the way overlapping labels are."""
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    noise = np.convolve(rng.standard_normal(n + 9), np.ones(10), mode="valid")[:n] * 0.001
    gross = planted * 0.001 * f1 + noise
    cost = 0.0013 + 0.0002 * np.abs(f2)
    return pd.DataFrame({
        "bar_start": np.arange(n) * 60, "f1": f1, "f2": f2,
        "gross_return": gross, "net_return": gross - cost,
    })


def _config(**overrides) -> StudyConfig:
    base = dict(
        horizons_minutes=[10], seeds=[0, 1], folds=3, test_share=0.5, validate_fraction=0.15,
        epochs=60, hidden=(8,), n_boot=60, n_perm=60, top_fractions=[0.1, 0.5], block_seed=0,
    )
    base.update(overrides)
    return StudyConfig(**base)


def test_a_planted_signal_is_found_and_flagged_significant():
    bars = _labeled_bars(6_000, planted=3.0, seed=1)
    rows, summary = run_horizon(bars, FEATURES, 10, _config(), StudyObserver())
    assert len(rows) == 2 * 3
    assert all(r["gross_correlation"] > 0.3 for r in rows)
    assert summary["pooled"]["gross_correlation"] > 0.3
    assert summary["pooled"]["p_value"] < 0.05
    assert summary["pooled"]["ci_low"] > 0.2
    assert summary["seed_correlations"]["mean"] > 0.3


def test_no_signal_gives_a_correlation_near_zero_and_an_interval_around_it():
    bars = _labeled_bars(6_000, planted=0.0, seed=2)
    _, summary = run_horizon(bars, FEATURES, 10, _config(), StudyObserver())
    pooled = summary["pooled"]
    assert abs(pooled["gross_correlation"]) < 0.1
    assert pooled["ci_low"] < 0 < pooled["ci_high"] or abs(pooled["gross_correlation"]) < 0.05


def test_top_slices_report_net_return_against_trading_every_bar():
    bars = _labeled_bars(6_000, planted=3.0, seed=3)
    _, summary = run_horizon(bars, FEATURES, 10, _config(), StudyObserver())
    slices = {s["top_fraction"]: s for s in summary["top_slices"]}
    assert set(slices) == {0.1, 0.5}
    assert slices[0.1]["mean_gross"] > summary["all_bars"]["mean_gross"]
    assert slices[0.1]["mean_net"] == pytest.approx(slices[0.1]["mean_gross"] - slices[0.1]["mean_cost"])
    assert slices[0.1]["n"] < slices[0.5]["n"]


def test_the_observer_sees_every_fold_seed_and_can_abort():
    bars = _labeled_bars(4_000, planted=1.0, seed=4)
    seen: list[dict] = []

    class Stop(Exception):
        pass

    def on_row(row: dict) -> None:
        seen.append(row)
        if len(seen) == 2:
            raise Stop

    with pytest.raises(Stop):
        run_horizon(bars, FEATURES, 10, _config(), StudyObserver(on_row=on_row))
    assert len(seen) == 2
    assert {"horizon_minutes", "seed", "fold", "n", "gross_correlation"} <= set(seen[0])
