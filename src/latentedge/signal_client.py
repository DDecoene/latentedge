from pathlib import Path

import numpy as np

from latentedge.model import load


class PredictionError(Exception):
    pass


class SignalClient:
    def __init__(self, model_path: Path, input_dim: int):
        self._model = load(model_path, input_dim=input_dim)

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if not np.isfinite(features).all():
            raise PredictionError("input feature matrix contains NaN/Inf")

        import mlx.core as mx

        predictions = np.array(self._model(mx.array(features.astype(np.float32))))

        if not np.isfinite(predictions).all():
            raise PredictionError("model produced NaN/Inf predictions")

        return predictions
