from pathlib import Path

import numpy as np

from latentedge.features import load_feature_stats, stats_to_arrays, target_stats, unstandardize_value
from latentedge.model import load


class PredictionError(Exception):
    pass


class SignalClient:
    def __init__(self, model_path: Path, input_dim: int, feature_columns: list[str] | None = None):
        self._model = load(model_path, input_dim=input_dim)
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._target_stats: tuple[float, float] | None = None

        if feature_columns is not None:
            stats_path = Path(str(model_path) + ".stats.json")
            if stats_path.exists():
                stats = load_feature_stats(stats_path)
                self._means, self._stds = stats_to_arrays(stats, feature_columns)
                # The model is trained on a standardized return (gross or
                # net), so its raw output is in standard deviations, not
                # return units. Consumers (the guard's sign test and
                # sizing) need real returns, so map the output back here,
                # once.
                self._target_stats = target_stats(stats) if ("target" in stats or "net_return" in stats) else None

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if not np.isfinite(features).all():
            raise PredictionError("input feature matrix contains NaN/Inf")

        if self._means is not None and self._stds is not None:
            features = (features - self._means) / self._stds

        import mlx.core as mx

        predictions = np.array(self._model(mx.array(features.astype(np.float32))))

        if self._target_stats is not None:
            predictions = unstandardize_value(predictions, self._target_stats)

        if not np.isfinite(predictions).all():
            raise PredictionError("model produced NaN/Inf predictions")

        return predictions
