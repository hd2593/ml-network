from __future__ import annotations


def choose_cwnd_action(probability: float, increase_threshold: float = 0.35, decrease_threshold: float = 0.65) -> str:
    if probability < increase_threshold:
        return "increase"
    if probability >= decrease_threshold:
        return "decrease"
    return "stay"


def apply_cwnd_action(cwnd: float, action: str, increase_factor: float = 1.08, decrease_factor: float = 0.75) -> float:
    cwnd = max(float(cwnd), 1.0)
    if action == "increase":
        return cwnd * increase_factor
    if action == "decrease":
        return max(1.0, cwnd * decrease_factor)
    return cwnd

