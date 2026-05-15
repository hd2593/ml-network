from __future__ import annotations

import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.datasets.schema import GROUP_COLUMNS
from .tcp_features import add_derived_tcp_features, numeric_feature_columns


def _log(msg: str) -> None:
    print(msg, flush=True)


def _vectorized_rolling_trend(y: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Slope of the best-fit line over a rolling window, fully vectorised.

    Pandas' ``rolling.apply(np.polyfit)`` does this in Python and is the single
    slowest part of the feature pipeline. On 3 M rows it adds minutes per
    feature. The closed-form slope for x = 0..window-1 lets us collapse that
    to one numpy pass.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window < 2 or n < min_periods:
        return out

    valid = np.isfinite(y).astype(np.float64)
    y_clean = np.where(np.isfinite(y), y, 0.0)
    idx = np.arange(n, dtype=np.float64)

    # Cumulative sums (prepend 0 so cum[a:b] gives sum(y[a:b]))
    cum_y = np.concatenate(([0.0], np.cumsum(y_clean)))
    cum_iy = np.concatenate(([0.0], np.cumsum(idx * y_clean)))
    cum_valid = np.concatenate(([0.0], np.cumsum(valid)))

    end = np.arange(min_periods - 1, n)
    start = np.maximum(0, end - window + 1)
    width = end - start + 1
    sum_y = cum_y[end + 1] - cum_y[start]
    sum_iy_global = cum_iy[end + 1] - cum_iy[start]
    sum_iy_local = sum_iy_global - start * sum_y

    mean_x = (width - 1) / 2.0
    # var(x) for x = 0..width-1, multiplied by width: width*(width^2 - 1)/12
    denom = width * (width * width - 1) / 12.0
    numerator = sum_iy_local - mean_x * sum_y
    valid_count = cum_valid[end + 1] - cum_valid[start]

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denom > 0, numerator / denom, np.nan)
    slope = np.where(valid_count >= min_periods, slope, np.nan)
    out[end] = slope
    return out


def _rolling_window_stats(y: np.ndarray, window: int, min_periods: int) -> dict[str, np.ndarray]:
    """Compute ``mean/std/min/max/trend`` over a rolling window in one shot."""
    series = pd.Series(y)
    rolling = series.rolling(window, min_periods=min_periods)
    return {
        "mean": rolling.mean().to_numpy(),
        "std": rolling.std().to_numpy(),
        "min": rolling.min().to_numpy(),
        "max": rolling.max().to_numpy(),
        "trend": _vectorized_rolling_trend(y, window, min_periods),
    }


def build_sliding_window_features(
    frame: pd.DataFrame,
    window_size: int = 8,
    progress_every: int = 200,
    label: str | None = None,
) -> pd.DataFrame:
    """Create tabular rolling features for tree models.

    Args:
        frame: labeled telemetry rows (output of ``add_congestion_label``).
        window_size: width of the rolling window.
        progress_every: emit a progress line after this many flows.
        label: optional tag included in progress lines (e.g. dataset name).
    """
    if frame.empty:
        return pd.DataFrame()

    tag = f"[{label}] " if label else ""
    _log(f"{tag}windowing: enriching derived TCP features (rows={len(frame):,})")
    enriched = add_derived_tcp_features(frame)
    enriched = enriched.sort_values(GROUP_COLUMNS + ["timestamp"]).copy()
    features = numeric_feature_columns(enriched)
    exclude = {"congestion"}
    features = [column for column in features if column not in exclude]
    min_periods = max(2, window_size // 2)

    grouped = enriched.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    total_flows = grouped.ngroups
    _log(f"{tag}windowing: {total_flows:,} flows, {len(features)} features, window={window_size}")

    pieces = []
    started = time.time()
    rows_done = 0
    for flow_idx, (_keys, group) in enumerate(grouped, start=1):
        base_cols = {col: group[col].to_numpy() for col in GROUP_COLUMNS + ["timestamp"]}
        if "congestion" in group.columns:
            base_cols["congestion"] = group["congestion"].astype(int).to_numpy()
        feature_cols: dict[str, np.ndarray] = {}
        for column in features:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            feature_cols[f"{column}_last"] = values
            stats = _rolling_window_stats(values, window_size, min_periods)
            for stat_name, stat_values in stats.items():
                feature_cols[f"{column}_{stat_name}"] = stat_values
        out = pd.DataFrame({**base_cols, **feature_cols})
        pieces.append(out)
        rows_done += len(group)
        if flow_idx % progress_every == 0 or flow_idx == total_flows:
            elapsed = time.time() - started
            rate = rows_done / elapsed if elapsed > 0 else 0.0
            _log(
                f"{tag}windowing: flows {flow_idx:,}/{total_flows:,} "
                f"rows {rows_done:,}/{len(frame):,} ({rate:,.0f} rows/s, "
                f"elapsed {elapsed:.1f}s)"
            )

    result = pd.concat(pieces, ignore_index=True)
    feature_cols = [column for column in result.columns if column not in GROUP_COLUMNS + ["timestamp", "congestion"]]
    result = result.dropna(subset=feature_cols, how="all")
    _log(f"{tag}windowing: kept {len(result):,} rows after dropna")
    return result


def _augment_with_engineered(
    values: np.ndarray,
    feature_names: list[str],
    engineered_window: int,
) -> tuple[np.ndarray, list[str]]:
    """Append causal rolling mean/std/min/max/trend per feature to ``values``.

    ``values`` is the ``[T, F]`` per-flow matrix. Stats are computed via the
    same ``_rolling_window_stats`` the tree pipeline uses, which is left-aligned
    (causal — at row t only rows [t - K + 1, t] are visible). NaNs from short
    prefixes are zero-filled *after* the rolling computation so they do not
    poison the rolling itself.
    """
    if values.size == 0:
        return values, list(feature_names)
    min_periods = max(2, engineered_window // 2)
    new_columns: list[np.ndarray] = [values]
    new_names: list[str] = [f"{name}_last" for name in feature_names]
    for j, name in enumerate(feature_names):
        stats = _rolling_window_stats(values[:, j], engineered_window, min_periods)
        for stat_name in ("mean", "std", "min", "max", "trend"):
            column = np.asarray(stats[stat_name], dtype=np.float32)
            column = np.where(np.isfinite(column), column, 0.0)
            new_columns.append(column.reshape(-1, 1))
            new_names.append(f"{name}_{stat_name}")
    augmented = np.concatenate(new_columns, axis=1).astype(np.float32, copy=False)
    return augmented, new_names


def build_sequence_tensors(
    frame: pd.DataFrame,
    window_size: int = 8,
    max_sequences: int | None = None,
    random_state: int = 42,
    label: str | None = None,
    with_engineered: bool = False,
    engineered_window: int = 8,
):
    """Return X, y, feature_names for LSTM training.

    Sequence windows are built with NumPy views per flow instead of one Python
    append per window. When ``max_sequences`` is set, global window indices are
    sampled before materialization so large runs do not first build millions of
    windows only to discard most of them.

    When ``with_engineered`` is True, each timestep is augmented with causal
    rolling statistics (mean/std/min/max/trend over the prior
    ``engineered_window`` samples per feature) using
    :func:`_rolling_window_stats`. This gives the LSTM the same per-row summary
    information :func:`build_sliding_window_features` produces for tree models.
    """
    if frame.empty:
        return np.empty((0, window_size, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    tag = f"[{label}] " if label else ""
    started = time.time()
    _log(f"{tag}sequences: enriching and sorting rows={len(frame):,}")
    enriched = add_derived_tcp_features(frame).sort_values(GROUP_COLUMNS + ["timestamp"]).copy()
    base_features = [column for column in numeric_feature_columns(enriched) if column != "congestion"]
    if not base_features:
        return np.empty((0, window_size, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), base_features

    enriched[base_features] = enriched[base_features].apply(pd.to_numeric, errors="coerce")
    grouped = enriched.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    group_sizes = grouped.size().to_numpy(dtype=np.int64)
    sequence_counts = np.maximum(0, group_sizes - window_size + 1)
    total_sequences = int(sequence_counts.sum())

    if with_engineered:
        feature_names_out = [f"{name}_last" for name in base_features]
        for name in base_features:
            for stat_name in ("mean", "std", "min", "max", "trend"):
                feature_names_out.append(f"{name}_{stat_name}")
        _log(
            f"{tag}sequences: with_engineered=True (engineered_window={engineered_window}); "
            f"{len(base_features)} base -> {len(feature_names_out)} per-step features"
        )
    else:
        feature_names_out = list(base_features)

    if total_sequences == 0:
        return (
            np.empty((0, window_size, len(feature_names_out)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            feature_names_out,
        )

    selected_global: np.ndarray | None = None
    if max_sequences is not None and max_sequences > 0 and total_sequences > max_sequences:
        rng = np.random.default_rng(random_state)
        selected_global = np.sort(rng.choice(total_sequences, size=max_sequences, replace=False))
        _log(f"{tag}sequences: sampling {max_sequences:,}/{total_sequences:,} windows before materialization")
    else:
        _log(f"{tag}sequences: materializing {total_sequences:,} windows")

    pieces: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    global_offset = 0
    rows_done = 0
    for flow_idx, ((_keys, group), count) in enumerate(zip(grouped, sequence_counts), start=1):
        if count <= 0:
            rows_done += len(group)
            continue

        local_indices = None
        if selected_global is not None:
            start = np.searchsorted(selected_global, global_offset, side="left")
            stop = np.searchsorted(selected_global, global_offset + count, side="left")
            if start == stop:
                global_offset += int(count)
                rows_done += len(group)
                continue
            local_indices = selected_global[start:stop] - global_offset

        values = (
            group[base_features]
            .ffill()
            .bfill()
            .fillna(0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        if with_engineered:
            values, _ = _augment_with_engineered(values, base_features, engineered_window)
        y = group["congestion"].fillna(0).astype(np.int64).to_numpy()

        windows = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=0)
        windows = np.moveaxis(windows, -1, 1)
        window_labels = y[window_size - 1 :]
        if local_indices is not None:
            windows = windows[local_indices]
            window_labels = window_labels[local_indices]

        pieces.append(np.ascontiguousarray(windows, dtype=np.float32))
        labels.append(np.asarray(window_labels, dtype=np.int64))
        global_offset += int(count)
        rows_done += len(group)

        if flow_idx % 500 == 0:
            elapsed = time.time() - started
            rate = rows_done / elapsed if elapsed > 0 else 0.0
            _log(f"{tag}sequences: flows {flow_idx:,}/{len(sequence_counts):,} rows {rows_done:,} ({rate:,.0f} rows/s)")

    if not pieces:
        return (
            np.empty((0, window_size, len(feature_names_out)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            feature_names_out,
        )

    X = np.concatenate(pieces, axis=0)
    y = np.concatenate(labels, axis=0)
    _log(f"{tag}sequences: built {len(X):,} windows in {time.time() - started:.1f}s")
    return X, y, feature_names_out
