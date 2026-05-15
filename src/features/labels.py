from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.datasets.schema import GROUP_COLUMNS


def _numeric_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_or_false(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def add_congestion_label(frame: pd.DataFrame, horizon: int = 2, label: str | None = None) -> pd.DataFrame:
    """Derive a near-future binary congestion label from observable symptoms."""
    if frame.empty:
        result = frame.copy()
        result["congestion"] = []
        return result

    tag = f"[{label}] " if label else ""
    started = time.time()
    print(f"{tag}labeling: {len(frame):,} rows, sorting...", flush=True)
    result = frame.sort_values(GROUP_COLUMNS + ["timestamp"]).copy()
    grouped = result.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    print(f"{tag}labeling: {grouped.ngroups:,} flows, computing rolling baselines", flush=True)

    rtt = _numeric_or_nan(result, "rtt_ms")
    delivery = _numeric_or_nan(result, "delivery_rate_bps")
    inflight = _numeric_or_nan(result, "in_flight")
    cwnd = _numeric_or_nan(result, "cwnd")
    delay = _numeric_or_nan(result, "one_way_delay_ms")

    rtt_baseline = rtt.groupby([result[c] for c in GROUP_COLUMNS], dropna=False, sort=False).transform(
        lambda s: s.rolling(20, min_periods=3).median()
    )
    delay_baseline = delay.groupby([result[c] for c in GROUP_COLUMNS], dropna=False, sort=False).transform(
        lambda s: s.rolling(20, min_periods=3).median()
    )
    delivery_prev = delivery.groupby([result[c] for c in GROUP_COLUMNS], dropna=False, sort=False).shift(horizon)

    pressure = (inflight / cwnd.replace(0, np.nan)) > 0.9
    rtt_inflation = rtt > (1.5 * rtt_baseline)
    delay_inflation = delay > (1.5 * delay_baseline)
    delivery_drop = delivery < (0.65 * pd.to_numeric(delivery_prev, errors="coerce"))
    loss = _bool_or_false(result, "loss_event")
    rebuffer = _bool_or_false(result, "rebuffer_event")

    symptom = (pressure | rtt_inflation | delay_inflation | delivery_drop | loss | rebuffer).fillna(False)
    print(f"{tag}labeling: propagating horizon-{horizon} future symptom across {grouped.ngroups:,} flows", flush=True)
    future_symptom = pd.Series(False, index=result.index)
    for flow_idx, (_, group) in enumerate(grouped, start=1):
        values = symptom.loc[group.index].astype(int)
        future = values.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
        future_symptom.loc[group.index] = future.fillna(False).astype(bool)
        if flow_idx % 5000 == 0:
            print(f"{tag}labeling: processed {flow_idx:,}/{grouped.ngroups:,} flows", flush=True)
    result["congestion"] = future_symptom.fillna(False).astype(int)
    elapsed = time.time() - started
    pos = int(result["congestion"].sum())
    print(
        f"{tag}labeling: done in {elapsed:.1f}s, positive rate {pos / max(1, len(result)):.3f} ({pos:,} positives)",
        flush=True,
    )
    return result
