from __future__ import annotations

from dataclasses import dataclass

from .cwnd_policy import apply_cwnd_action, choose_cwnd_action


@dataclass
class RateDecision:
    probability: float
    action: str
    old_cwnd: float
    new_cwnd: float


class MLRateController:
    def __init__(self, model, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns

    def decide(self, feature_row, current_cwnd: float) -> RateDecision:
        X = feature_row[self.feature_columns].to_frame().T.fillna(0)
        probability = float(self.model.predict_proba(X)[0, 1]) if hasattr(self.model, "predict_proba") else float(self.model.predict(X)[0])
        action = choose_cwnd_action(probability)
        new_cwnd = apply_cwnd_action(current_cwnd, action)
        return RateDecision(probability, action, current_cwnd, new_cwnd)

