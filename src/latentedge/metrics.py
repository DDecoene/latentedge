"""Training metrics: how well a trained model actually predicts, on data
it never trained on — persisted to disk so a TUI run's numbers survive
past the terminal scrolling away.
"""

import json
from pathlib import Path
from typing import NotRequired, TypedDict

import mlx.core as mx
import numpy as np

from latentedge.features import target_stats, unstandardize_value
from latentedge.model import NetReturnRegressor
from latentedge.training_data import AssembledTrainingData, SplitArrays


class SplitMetrics(TypedDict):
    n: int
    mse: float
    baseline_mse: float
    correlation: float
    # Only when the pre-cost return is known; see prediction_correlations.
    net_correlation: NotRequired[float]
    gross_correlation: NotRequired[float]
    cost_correlation: NotRequired[float]
    gross_std: NotRequired[float]
    cost_std: NotRequired[float]


class TrainingMetrics(TypedDict):
    epochs: int
    final_loss: float
    loss_history: list[float]
    splits: dict[str, SplitMetrics]
    test_start: NotRequired[int]
    feature_columns: NotRequired[list[str]]
    target: NotRequired[str]
    hidden_sizes: NotRequired[list[int]]
    label_horizon_seconds: NotRequired[int]
    label_barrier_stds: NotRequired[float]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return float("nan")


def prediction_correlations(predictions: np.ndarray, net: np.ndarray, gross: np.ndarray) -> dict[str, float]:
    """Does a model that predicts net return predict the direction of the
    move, or only what trading costs? net = gross - cost, so correlate the
    predictions with each part. A prediction that tracks the cost part
    (cost_correlation clearly negative, gross_correlation near zero) has
    learned gas and volatility, not price direction. The two spreads say
    how much room each part has to explain the net correlation: a cost that
    barely varies cannot account for much of it."""
    cost = gross - net
    return {
        "gross_correlation": _correlation(predictions, gross),
        "cost_correlation": _correlation(predictions, cost),
        "gross_std": float(np.std(gross)),
        "cost_std": float(np.std(cost)),
    }


def evaluate_predictions(
    predictions: np.ndarray, targets: np.ndarray, baseline_prediction: float, gross: np.ndarray | None = None,
    net: np.ndarray | None = None,
) -> SplitMetrics:
    """MSE against a trivial "always predict this constant" baseline, plus
    correlation — a model with MSE worse than the baseline, or near-zero
    correlation, hasn't learned a usable signal regardless of the raw
    loss number."""
    mse = float(np.mean((predictions - targets) ** 2))
    baseline_mse = float(np.mean((targets - baseline_prediction) ** 2))

    result: SplitMetrics = {
        "n": len(targets), "mse": mse, "baseline_mse": baseline_mse,
        "correlation": _correlation(predictions, targets),
    }
    if gross is not None:
        actual_net = net if net is not None else targets
        result["net_correlation"] = _correlation(predictions, actual_net)
        result.update(prediction_correlations(predictions, actual_net, gross))  # type: ignore[typeddict-item]
    return result


def evaluate_model(model: NetReturnRegressor, assembled: AssembledTrainingData) -> dict[str, SplitMetrics]:
    # The model was trained to predict its target (net or gross return) on
    # a standardized scale (see cli.train / TrainScreen._run_train) —
    # unstandardize its raw output back to real return units before
    # comparing against SplitArrays.y, which is always on the raw scale, so
    # MSE here reads in the same units as the returns it's meant to predict.
    stats_for_target = target_stats(assembled.stats)
    baseline_prediction = float(np.mean(assembled.train.y))
    splits: dict[str, SplitArrays] = {"train": assembled.train, "validate": assembled.validate, "test": assembled.test}
    return {
        name: evaluate_predictions(
            unstandardize_value(np.array(model(mx.array(split.x))), stats_for_target), split.y, baseline_prediction,
            gross=split.gross, net=split.net,
        )
        for name, split in splits.items()
    }


def build_training_metrics(
    model: NetReturnRegressor, assembled: AssembledTrainingData, losses: list[float]
) -> TrainingMetrics:
    metrics: TrainingMetrics = {
        "epochs": len(losses),
        "final_loss": losses[-1],
        "loss_history": losses,
        "splits": evaluate_model(model, assembled),
    }
    if assembled.test_start is not None:
        metrics["test_start"] = assembled.test_start
    metrics["target"] = assembled.target
    metrics["hidden_sizes"] = list(assembled.hidden_sizes)
    metrics["feature_columns"] = list(assembled.feature_columns)
    metrics["label_horizon_seconds"] = assembled.label_settings.horizon_seconds
    metrics["label_barrier_stds"] = assembled.label_settings.barrier_stds
    return metrics


def save_training_metrics(metrics: TrainingMetrics, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2))


def load_training_metrics(path: Path) -> TrainingMetrics:
    result: TrainingMetrics = json.loads(path.read_text())
    return result
