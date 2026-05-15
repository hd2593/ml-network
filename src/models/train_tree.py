from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .evaluate import evaluate_classifier, feature_target_split


@dataclass
class FeatureAlignedModel:
    estimator: object
    feature_columns: list[str]

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        aligned = X.copy()
        for column in self.feature_columns:
            if column not in aligned.columns:
                aligned[column] = 0.0
        aligned = aligned[self.feature_columns]
        return aligned.replace([np.inf, -np.inf], np.nan).fillna(0)

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(self._align(X))

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(self._align(X))

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_", None)


DEFAULT_RF_TRAIN_CAP = 2_000_000


def train_random_forest(train_frame: pd.DataFrame, max_train_rows: int | None = DEFAULT_RF_TRAIN_CAP, **kwargs):
    if max_train_rows is not None and len(train_frame) > max_train_rows:
        original = len(train_frame)
        train_frame = train_frame.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
        print(
            f"[random_forest] sub-sampled {original:,} -> {len(train_frame):,} rows "
            f"(set --max-train-rows higher or 0 to disable)",
            flush=True,
        )

    X, y, feature_columns = feature_target_split(train_frame)
    params = {
        "n_estimators": 300,
        "max_depth": 24,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 4,
        "verbose": 1,
    }
    params.update({key: value for key, value in kwargs.items() if value is not None})
    print(
        f"[random_forest] training on {len(X):,} rows x {len(feature_columns)} features "
        f"(n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
        f"n_jobs={params['n_jobs']})",
        flush=True,
    )
    started = time.time()
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    print(f"[random_forest] fit done in {time.time() - started:.1f}s", flush=True)
    return FeatureAlignedModel(model, feature_columns)


def _resolve_xgb_device(preferred: str | None) -> str:
    """Pick the XGBoost device string. Falls back to CPU if CUDA is unusable."""
    if preferred in (None, "auto"):
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return preferred


def train_xgboost(
    train_frame: pd.DataFrame,
    max_train_rows: int | None = DEFAULT_RF_TRAIN_CAP,
    device: str | None = None,
    **kwargs,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is optional. Install it or use --model random_forest.") from exc
    if max_train_rows is not None and len(train_frame) > max_train_rows:
        original = len(train_frame)
        train_frame = train_frame.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
        print(
            f"[xgboost] sub-sampled {original:,} -> {len(train_frame):,} rows "
            f"(set --max-train-rows higher or 0 to disable)",
            flush=True,
        )
    X, y, feature_columns = feature_target_split(train_frame)
    resolved_device = _resolve_xgb_device(device)
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": resolved_device,
        "n_jobs": -1,
        "verbosity": 1,
    }
    params.update({key: value for key, value in kwargs.items() if value is not None})
    print(
        f"[xgboost] training on {len(X):,} rows x {len(feature_columns)} features "
        f"(device={params['device']}, tree_method={params['tree_method']}, "
        f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']})",
        flush=True,
    )
    started = time.time()
    model = XGBClassifier(**params)
    model.fit(X, y)
    print(f"[xgboost] fit done in {time.time() - started:.1f}s", flush=True)
    return FeatureAlignedModel(model, feature_columns)


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    return joblib.load(path)
