"""Training metrics: how well a trained model actually predicts, on data
it never trained on — persisted to disk so a TUI run's numbers survive
past the terminal scrolling away.
"""

import json
from pathlib import Path
from typing import TypedDict

import mlx.core as mx
import numpy as np

from latentedge.model import NetReturnRegressor
from latentedge.training_data import AssembledTrainingData, SplitArrays


class SplitMetrics(TypedDict):
    n: int
    mse: float
    baseline_mse: float
    correlation: float


class TrainingMetrics(TypedDict):
    epochs: int
    final_loss: float
    loss_history: list[float]
    splits: dict[str, SplitMetrics]


def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray, baseline_prediction: float) -> SplitMetrics:
    """MSE against a trivial "always predict this constant" baseline, plus
    correlation — a model with MSE worse than the baseline, or near-zero
    correlation, hasn't learned a usable signal regardless of the raw
    loss number."""
    mse = float(np.mean((predictions - targets) ** 2))
    baseline_mse = float(np.mean((targets - baseline_prediction) ** 2))

    if len(targets) > 1 and np.std(predictions) > 0 and np.std(targets) > 0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    else:
        correlation = float("nan")

    return {"n": len(targets), "mse": mse, "baseline_mse": baseline_mse, "correlation": correlation}


def evaluate_model(model: NetReturnRegressor, assembled: AssembledTrainingData) -> dict[str, SplitMetrics]:
    baseline_prediction = float(np.mean(assembled.train.y))
    splits: dict[str, SplitArrays] = {"train": assembled.train, "validate": assembled.validate, "test": assembled.test}
    return {
        name: evaluate_predictions(np.array(model(mx.array(split.x))), split.y, baseline_prediction)
        for name, split in splits.items()
    }


def build_training_metrics(
    model: NetReturnRegressor, assembled: AssembledTrainingData, losses: list[float]
) -> TrainingMetrics:
    return {
        "epochs": len(losses),
        "final_loss": losses[-1],
        "loss_history": losses,
        "splits": evaluate_model(model, assembled),
    }


def save_training_metrics(metrics: TrainingMetrics, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2))


def load_training_metrics(path: Path) -> TrainingMetrics:
    result: TrainingMetrics = json.loads(path.read_text())
    return result
