from __future__ import annotations

import os
import json
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CrowdPredictor:
    def __init__(self, model_path: str, window_size: int = 24, horizon_steps: int = 15):
        self.model_path = model_path
        self.window_size = window_size
        self.horizon_steps = horizon_steps
        self.action_feature_size = 4
        self.attribute_feature_size = 4
        self.feature_version = 2
        self.pipeline: Pipeline | None = None
        self.metrics = {}
        self._load_if_available()

    def _load_if_available(self) -> None:
        meta_path = self._meta_path()
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.window_size = int(meta.get("window_size", self.window_size))
            self.horizon_steps = int(meta.get("horizon_steps", self.horizon_steps))
            self.feature_version = int(meta.get("feature_version", 1))
            self.action_feature_size = int(meta.get("action_feature_size", self.action_feature_size))
            self.attribute_feature_size = int(meta.get("attribute_feature_size", self.attribute_feature_size))

        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)

    def _meta_path(self) -> str:
        model_dir = os.path.dirname(self.model_path)
        return os.path.join(model_dir, "crowd_predictor_meta.json")

    def is_trained(self) -> bool:
        return self.pipeline is not None

    def _legacy_feature_size(self) -> int:
        return self.window_size * 2 + 1

    def _v2_feature_size(self) -> int:
        return self.window_size * 2 + 1 + self.action_feature_size + self.attribute_feature_size

    def _resolved_feature_version(self) -> int:
        if self.pipeline is None:
            return self.feature_version

        n_features = int(getattr(self.pipeline, "n_features_in_", 0) or 0)
        if n_features == self._legacy_feature_size():
            return 1
        if n_features == self._v2_feature_size():
            return 2
        return self.feature_version

    @staticmethod
    def _normalize_feature_vector(values: Optional[Sequence[float]], expected_len: int) -> List[float]:
        if expected_len <= 0:
            return []
        if values is None:
            return [0.0] * expected_len

        out = [float(v) for v in values[:expected_len]]
        if len(out) < expected_len:
            out.extend([0.0] * (expected_len - len(out)))
        return out

    def _build_feature_vector(
        self,
        count_window: Sequence[float],
        density_window: Sequence[float],
        elapsed_value: float,
        action_features: Optional[Sequence[float]] = None,
        attribute_features: Optional[Sequence[float]] = None,
        feature_version: Optional[int] = None,
    ) -> List[float]:
        version = self.feature_version if feature_version is None else int(feature_version)
        features = [float(v) for v in count_window] + [float(v) for v in density_window] + [float(elapsed_value)]

        if version >= 2:
            features.extend(self._normalize_feature_vector(action_features, self.action_feature_size))
            features.extend(self._normalize_feature_vector(attribute_features, self.attribute_feature_size))

        return features

    def _build_training_rows(
        self,
        count_series: Sequence[int],
        density_series: Sequence[float],
        elapsed_series: Sequence[float],
        action_series: Optional[Sequence[Sequence[float]]] = None,
        attribute_series: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        min_len = min(len(count_series), len(density_series), len(elapsed_series))
        if action_series is not None:
            min_len = min(min_len, len(action_series))
        if attribute_series is not None:
            min_len = min(min_len, len(attribute_series))

        feature_version = self.feature_version
        expected_size = self._legacy_feature_size() if feature_version <= 1 else self._v2_feature_size()
        if min_len < self.window_size + self.horizon_steps:
            return np.empty((0, expected_size), dtype=np.float32), np.empty((0,), dtype=np.float32)

        x_rows: List[List[float]] = []
        y_rows: List[float] = []

        max_start = min_len - self.window_size - self.horizon_steps + 1
        for start in range(max_start):
            end = start + self.window_size
            target_idx = end + self.horizon_steps - 1

            count_window = [float(v) for v in count_series[start:end]]
            density_window = [float(v) for v in density_series[start:end]]
            elapsed_value = float(elapsed_series[end - 1])
            action_features = action_series[end - 1] if action_series is not None else None
            attribute_features = attribute_series[end - 1] if attribute_series is not None else None

            features = self._build_feature_vector(
                count_window,
                density_window,
                elapsed_value,
                action_features=action_features,
                attribute_features=attribute_features,
                feature_version=feature_version,
            )
            x_rows.append(features)
            y_rows.append(float(count_series[target_idx]))

        return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)

    def fit_from_series_list(
        self,
        series_list: Iterable[dict],
        candidate_configs: Optional[Sequence[dict]] = None,
        shuffle_split: bool = True,
        random_state: int = 42,
        selection_target: str = "composite",
        incoming_threshold_ratio: float = 0.12,
        incoming_threshold_min: int = 2,
        max_rows_per_series: int = 0,
    ) -> dict:
        self.feature_version = 2
        x_parts = []
        y_parts = []
        rng_series = np.random.RandomState(random_state + 1000)
        for series in series_list:
            x_chunk, y_chunk = self._build_training_rows(
                series["counts"],
                series["densities"],
                series["elapsed"],
                series.get("actions"),
                series.get("attributes"),
            )
            if max_rows_per_series > 0 and len(x_chunk) > max_rows_per_series:
                keep_idx = rng_series.choice(len(x_chunk), size=int(max_rows_per_series), replace=False)
                x_chunk = x_chunk[keep_idx]
                y_chunk = y_chunk[keep_idx]
            if len(x_chunk) > 0:
                x_parts.append(x_chunk)
                y_parts.append(y_chunk)

        if not x_parts:
            raise ValueError(
                "Not enough samples to train. Provide longer count series or reduce window/horizon."
            )

        x = np.vstack(x_parts)
        y = np.hstack(y_parts)

        if shuffle_split:
            rng = np.random.RandomState(random_state)
            order = rng.permutation(len(x))
            x = x[order]
            y = y[order]

        split_idx = int(len(x) * 0.8)
        if split_idx <= 0 or split_idx >= len(x):
            raise ValueError("Need more training samples to create train/validation split.")

        x_train = x[:split_idx]
        y_train = y[:split_idx]
        x_val = x[split_idx:]
        y_val = y[split_idx:]

        if candidate_configs is None:
            candidate_configs = [
                {
                    "name": "balanced_default",
                    "hidden_layer_sizes": (96, 48),
                    "alpha": 0.0005,
                    "learning_rate_init": 0.0010,
                    "max_iter": 1000,
                },
                {
                    "name": "wider_context",
                    "hidden_layer_sizes": (128, 64),
                    "alpha": 0.0008,
                    "learning_rate_init": 0.0008,
                    "max_iter": 1200,
                },
                {
                    "name": "regularized_stable",
                    "hidden_layer_sizes": (96, 64, 32),
                    "alpha": 0.0012,
                    "learning_rate_init": 0.0007,
                    "max_iter": 1300,
                },
            ]

        leaderboard = []
        best_score = None
        best_pipeline = None
        best_result = None

        for idx, cfg in enumerate(candidate_configs):
            local_seed = int(random_state + idx)
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "nn",
                        MLPRegressor(
                            hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", (96, 48))),
                            activation="relu",
                            solver="adam",
                            alpha=float(cfg.get("alpha", 0.0005)),
                            learning_rate_init=float(cfg.get("learning_rate_init", 0.0010)),
                            max_iter=int(cfg.get("max_iter", 1000)),
                            random_state=local_seed,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=25,
                        ),
                    ),
                ]
            )

            pipeline.fit(x_train, y_train)
            val_pred = np.maximum(pipeline.predict(x_val), 0)

            mae = float(mean_absolute_error(y_val, val_pred))
            rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
            mape = float(np.mean(np.abs((y_val - val_pred) / np.maximum(y_val, 1.0))) * 100.0)
            current_counts = x_val[:, self.window_size - 1]
            delta_true = y_val - current_counts
            delta_pred = val_pred - current_counts
            incoming_threshold = np.maximum(float(incoming_threshold_min), current_counts * float(incoming_threshold_ratio))
            incoming_true = delta_true >= incoming_threshold
            incoming_pred = delta_pred >= incoming_threshold
            incoming_accuracy = float(np.mean(incoming_true == incoming_pred) * 100.0)

            # Composite score keeps MAE primary while penalizing instability and relative error.
            if selection_target == "incoming_accuracy":
                # Lower score is better; negative incoming accuracy allows maximization with tie-breakers.
                score = (-incoming_accuracy) + 0.02 * mae + 0.01 * rmse
            elif selection_target == "mape":
                score = mape + 0.08 * mae + 0.02 * rmse
            else:
                score = mae + 0.15 * rmse + 0.01 * mape
            row = {
                "name": str(cfg.get("name", f"candidate_{idx + 1}")),
                "hidden_layer_sizes": list(tuple(cfg.get("hidden_layer_sizes", (96, 48)))),
                "alpha": float(cfg.get("alpha", 0.0005)),
                "learning_rate_init": float(cfg.get("learning_rate_init", 0.0010)),
                "max_iter": int(cfg.get("max_iter", 1000)),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape_percent": round(mape, 4),
                "incoming_accuracy_percent": round(incoming_accuracy, 4),
                "score": round(score, 4),
            }
            leaderboard.append(row)

            if best_score is None or score < best_score:
                best_score = score
                best_pipeline = pipeline
                best_result = row

        if best_pipeline is None or best_result is None:
            raise RuntimeError("Model selection failed: no valid candidate was trained.")

        self.pipeline = best_pipeline
        mae = float(best_result["mae"])
        rmse = float(best_result["rmse"])
        mape = float(best_result["mape_percent"])

        self.metrics = {
            "samples": int(len(x)),
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "mape_percent": round(mape, 3),
            "incoming_accuracy_percent": round(float(best_result["incoming_accuracy_percent"]), 3),
            "selection_score": round(float(best_result["score"]), 3),
            "selection_target": selection_target,
            "best_config_name": best_result["name"],
            "best_hidden_layers": "-".join(str(v) for v in best_result["hidden_layer_sizes"]),
            "best_alpha": round(float(best_result["alpha"]), 6),
            "best_learning_rate_init": round(float(best_result["learning_rate_init"]), 6),
            "best_max_iter": int(best_result["max_iter"]),
            "candidate_count": int(len(leaderboard)),
            "shuffle_split": int(bool(shuffle_split)),
            "max_rows_per_series": int(max_rows_per_series),
        }
        self.metrics["selection_leaderboard"] = leaderboard
        return self.metrics

    def save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save: model is not trained.")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "window_size": self.window_size,
                    "horizon_steps": self.horizon_steps,
                    "feature_version": self.feature_version,
                    "action_feature_size": self.action_feature_size,
                    "attribute_feature_size": self.attribute_feature_size,
                },
                f,
                indent=2,
            )

    def predict(
        self,
        count_history: Sequence[int],
        density_history: Sequence[float],
        elapsed_ratio: float,
        action_features: Optional[Sequence[float]] = None,
        attribute_features: Optional[Sequence[float]] = None,
    ) -> int:
        if self.pipeline is None:
            raise RuntimeError("Prediction model is not trained or loaded.")

        if len(count_history) < self.window_size or len(density_history) < self.window_size:
            return int(count_history[-1]) if count_history else 0

        count_window = np.array(count_history[-self.window_size :], dtype=np.float32)
        density_window = np.array(density_history[-self.window_size :], dtype=np.float32)
        feature_version = self._resolved_feature_version()
        features = self._build_feature_vector(
            count_window,
            density_window,
            float(elapsed_ratio),
            action_features=action_features,
            attribute_features=attribute_features,
            feature_version=feature_version,
        )
        latest_window = np.array(features, dtype=np.float32).reshape(1, -1)

        pred = float(self.pipeline.predict(latest_window)[0])
        return max(0, int(round(pred)))

    def predict_with_context_fallback(
        self,
        count_history: Sequence[int],
        density_history: Sequence[float],
        elapsed_ratio: float,
        fallback_fn: Callable[[Sequence[int]], int],
        warmup_ready: bool,
        action_features: Optional[Sequence[float]] = None,
        attribute_features: Optional[Sequence[float]] = None,
    ) -> int:
        if not warmup_ready or self.pipeline is None:
            return int(fallback_fn(count_history))
        return self.predict(
            count_history,
            density_history,
            elapsed_ratio,
            action_features=action_features,
            attribute_features=attribute_features,
        )
