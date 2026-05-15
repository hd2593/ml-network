"""Flow-level train/val/test splits shared by the tree and LSTM pipelines.

The previous tree pipeline split *windowed rows* with sklearn's
``train_test_split``. Because adjacent sliding windows from the same flow
overlap by ``window_size - 1`` raw samples, that split leaked temporally
correlated rows from train into val/test and inflated tree metrics. The LSTM
pipeline always split at the flow level, so the two models were not being
judged on comparable held-out data.

This module gives both pipelines one helper so the split is identical, and
optionally persists the chosen flow ids to ``flow_splits.json`` so downstream
scripts can re-apply the same assignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.schema import GROUP_COLUMNS


def flow_split(
    frame: pd.DataFrame,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition ``frame`` so that no flow appears in both returned frames."""
    flow_keys = frame[GROUP_COLUMNS].drop_duplicates().reset_index(drop=True)
    if len(flow_keys) <= 1:
        return frame, frame.iloc[0:0]
    rng = np.random.default_rng(seed)
    permuted = flow_keys.sample(
        frac=1.0, random_state=int(rng.integers(0, 1 << 31))
    ).reset_index(drop=True)
    cut = max(1, int(round(len(permuted) * (1.0 - val_size))))
    train_keys = permuted.iloc[:cut]
    val_keys = permuted.iloc[cut:]
    train_frame = frame.merge(train_keys, on=GROUP_COLUMNS, how="inner")
    val_frame = frame.merge(val_keys, on=GROUP_COLUMNS, how="inner")
    return train_frame, val_frame


def three_way_flow_split(
    frame: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage flow-level split: carve val first, then test from the remainder.

    Matches the previous LSTM behaviour: val taken at fraction ``val_size`` with
    ``seed``; test taken at fraction ``test_size`` from the remaining frame with
    ``seed + 1``.
    """
    train_plus_test, val = flow_split(frame, val_size=val_size, seed=seed)
    train, test = flow_split(train_plus_test, val_size=test_size, seed=seed + 1)
    return train, val, test


def _flow_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return (
        frame[GROUP_COLUMNS]
        .drop_duplicates()
        .astype({col: "object" for col in GROUP_COLUMNS})
        .to_dict(orient="records")
    )


def write_flow_split_manifest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_path: str | Path,
    *,
    seed: int,
    val_size: float,
    test_size: float,
) -> Path:
    """Dump the unique flow ids for each split to ``out_path`` (JSON)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "group_columns": GROUP_COLUMNS,
        "seed": int(seed),
        "val_size": float(val_size),
        "test_size": float(test_size),
        "train": _flow_records(train),
        "val": _flow_records(val),
        "test": _flow_records(test),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def load_flow_split_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_manifest(
    frame: pd.DataFrame,
    manifest: dict,
    split: str,
) -> pd.DataFrame:
    """Inner-merge ``frame`` against the manifest's flow ids for ``split``."""
    if frame.empty:
        return frame
    records = manifest.get(split, [])
    if not records:
        return frame.iloc[0:0]
    keys = pd.DataFrame.from_records(records)
    for col in GROUP_COLUMNS:
        if col not in keys.columns:
            raise KeyError(f"manifest split '{split}' missing column '{col}'")
        keys[col] = keys[col].astype(frame[col].dtype, copy=False)
    return frame.merge(keys[GROUP_COLUMNS], on=GROUP_COLUMNS, how="inner")
