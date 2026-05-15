from __future__ import annotations

import numpy as np


def jains_fairness_index(throughputs) -> float:
    values = np.asarray(list(throughputs), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.square(values).sum() == 0:
        return 0.0
    return float(np.square(values.sum()) / (len(values) * np.square(values).sum()))


def reaction_latency_seconds(timestamps, probabilities, threshold: float = 0.65) -> float | None:
    for timestamp, probability in zip(timestamps, probabilities):
        if probability >= threshold:
            return float(timestamp)
    return None

