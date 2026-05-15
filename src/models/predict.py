from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.controller.cwnd_policy import choose_cwnd_action
from src.models.evaluate import feature_target_split
from src.models.train_tree import load_model


def predict_frame(model_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    model = load_model(model_path)
    X, _, _ = feature_target_split(frame.assign(congestion=0) if "congestion" not in frame.columns else frame)
    result = frame.copy()
    if hasattr(model, "predict_proba"):
        result["congestion_probability"] = model.predict_proba(X)[:, 1]
    else:
        result["congestion_probability"] = model.predict(X)
    result["cwnd_action"] = result["congestion_probability"].map(choose_cwnd_action)
    return result

