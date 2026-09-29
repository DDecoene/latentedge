import json
from pathlib import Path

import numpy as np

from latentedge.metrics import (
    build_training_metrics, evaluate_predictions, load_training_metrics, prediction_correlations, save_training_metrics,
)
from latentedge.model import NetReturnRegressor, train
from latentedge.training_data import AssembledTrainingData, SplitArrays


def test_evaluate_predictions_perfect_fit_has_zero_mse_and_unit_correlation():
    targets = np.array([0.1, -0.2, 0.3, -0.4], dtype="float32")
    result = evaluate_predictions(targets, targets, baseline_prediction=0.0)

    assert result["mse"] == 0.0
    assert result["correlation"] == 1.0
    assert result["n"] == 4


def test_evaluate_predictions_worse_than_baseline_is_visible_in_the_numbers():
    targets = np.array([1.0, -1.0, 1.0, -1.0], dtype="float32")
    wild_predictions = np.array([10.0, -10.0, -10.0, 10.0], dtype="float32")
    result = evaluate_predictions(wild_predictions, targets, baseline_prediction=0.0)

    assert result["mse"] > result["baseline_mse"]


def test_evaluate_predictions_handles_zero_variance_without_crashing():
    targets = np.array([0.5], dtype="float32")
    result = evaluate_predictions(targets, targets, baseline_prediction=0.5)

    assert np.isnan(result["correlation"])


def test_build_training_metrics_reports_all_three_splits():
    rng = np.random.default_rng(0)

    def make_split(n: int) -> SplitArrays:
        return SplitArrays(x=rng.normal(size=(n, 3)).astype("float32"), y=rng.normal(size=(n,)).astype("float32"))

    assembled = AssembledTrainingData(
        train=make_split(20), validate=make_split(8), test=make_split(8), input_dim=3,
        stats={"net_return": (0.0, 1.0)},
    )
    model = NetReturnRegressor(input_dim=3)
    losses = train(model, assembled.train.x, assembled.train.y, epochs=3, learning_rate=0.01)

    metrics = build_training_metrics(model, assembled, losses)

    assert metrics["epochs"] == 3
    assert metrics["final_loss"] == losses[-1]
    assert metrics["loss_history"] == losses
    assert set(metrics["splits"]) == {"train", "validate", "test"}
    for split_metrics in metrics["splits"].values():
        assert {"n", "mse", "baseline_mse", "correlation"} <= set(split_metrics)


def test_save_and_load_training_metrics_round_trip(tmp_path: Path):
    metrics = {"epochs": 5, "final_loss": 0.01, "loss_history": [0.1, 0.05, 0.01], "splits": {"train": {"n": 3}}}
    path = tmp_path / "model.safetensors.metrics.json"

    save_training_metrics(metrics, path)
    loaded = load_training_metrics(path)

    assert loaded == metrics
    assert json.loads(path.read_text()) == metrics


def test_a_prediction_that_tracks_only_cost_correlates_with_cost_not_direction():
    rng = np.random.default_rng(0)
    gross = rng.normal(0, 0.004, size=5000)
    cost = rng.uniform(0.001, 0.003, size=5000)
    net = gross - cost

    result = prediction_correlations(-cost, net, gross)

    assert result["cost_correlation"] < -0.99
    assert abs(result["gross_correlation"]) < 0.05
    assert result["gross_std"] > result["cost_std"] > 0


def test_a_prediction_that_tracks_direction_correlates_with_gross():
    rng = np.random.default_rng(1)
    gross = rng.normal(0, 0.004, size=5000)
    net = gross - 0.0013

    result = prediction_correlations(gross, net, gross)

    assert result["gross_correlation"] > 0.99
    assert abs(result["cost_correlation"]) < 0.05


def test_evaluate_predictions_reports_gross_metrics_only_when_gross_is_given():
    targets = np.array([0.1, -0.2, 0.3, -0.4])

    assert "gross_correlation" not in evaluate_predictions(targets, targets, 0.0)
    assert "gross_correlation" in evaluate_predictions(targets, targets, 0.0, gross=targets + 0.1)


def test_training_metrics_record_the_features_and_label_settings_and_gross_correlation():
    from latentedge.labeling import LabelSettings

    rng = np.random.default_rng(0)

    def make_split(n: int) -> SplitArrays:
        return SplitArrays(
            x=rng.normal(size=(n, 2)).astype("float32"), y=rng.normal(size=(n,)).astype("float32"),
            gross=rng.normal(size=(n,)),
        )

    assembled = AssembledTrainingData(
        train=make_split(20), validate=make_split(8), test=make_split(8), input_dim=2,
        stats={"net_return": (0.0, 1.0)}, feature_columns=("return_5", "return_15"),
        label_settings=LabelSettings(horizon_seconds=14400, barrier_stds=6.0),
    )
    model = NetReturnRegressor(input_dim=2)

    metrics = build_training_metrics(model, assembled, [0.5])

    assert metrics["feature_columns"] == ["return_5", "return_15"]
    assert metrics["label_horizon_seconds"] == 14400 and metrics["label_barrier_stds"] == 6.0
    assert all("gross_correlation" in m for m in metrics["splits"].values())
