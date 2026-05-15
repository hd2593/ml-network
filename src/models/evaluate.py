from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def feature_target_split(frame: pd.DataFrame):
    drop_cols = {"congestion", "source", "scenario_id", "flow_id", "timestamp"}
    feature_cols = [
        column
        for column in frame.columns
        if column not in drop_cols and pd.api.types.is_numeric_dtype(frame[column])
    ]
    X = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = frame["congestion"].fillna(0).astype(int)
    return X, y, feature_cols


def evaluate_classifier(model, frame: pd.DataFrame) -> dict:
    X, y, _ = feature_target_split(frame)
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = pred
    metrics = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    metrics["auc"] = float(roc_auc_score(y, prob)) if y.nunique() > 1 else None
    stable = y.eq(0)
    metrics["false_positive_rate_stable"] = float((pd.Series(pred, index=y.index)[stable] == 1).mean()) if stable.any() else 0.0
    return metrics


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

