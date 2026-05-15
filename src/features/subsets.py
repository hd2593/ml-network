"""Feature-subset selection for cross-source training.

This module computes the shared feature columns: numeric features with at
least one non-zero, finite value in every input table. That keeps training on
the observable surface common to the selected datasets.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.datasets.schema import GROUP_COLUMNS

NON_FEATURE_COLUMNS = set(GROUP_COLUMNS) | {"timestamp", "congestion"}


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def populated_feature_columns(frame: pd.DataFrame) -> set[str]:
    """Return numeric feature columns that contain at least one non-zero value."""
    populated: set[str] = set()
    for column in numeric_feature_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if np.any(np.isfinite(values) & (values != 0)):
            populated.add(column)
    return populated


def shared_feature_columns(frames: Iterable[pd.DataFrame]) -> list[str]:
    """Columns populated in every supplied frame, ordered for stable model layout."""
    populated_sets = [populated_feature_columns(frame) for frame in frames if not frame.empty]
    if not populated_sets:
        return []
    shared = set.intersection(*populated_sets)
    # Use the column order of the first frame so models stay deterministic.
    first = next(iter(frame for frame in frames if not frame.empty))
    return [column for column in numeric_feature_columns(first) if column in shared]


def restrict_to_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Drop feature columns not in ``feature_columns`` while preserving identifiers."""
    keep = [column for column in frame.columns if column in NON_FEATURE_COLUMNS or column in feature_columns]
    return frame[keep].copy()
