from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = [
    "source",
    "scenario_id",
    "flow_id",
    "timestamp",
    "rtt_ms",
    "min_rtt_ms",
    "delivery_rate_bps",
    "throughput_bps",
    "cwnd",
    "in_flight",
    "bytes_sent",
    "bytes_acked",
    "bytes_retrans",
    "segments_out",
    "segments_in",
    "packet_size_bytes",
    "one_way_delay_ms",
    "ack_spacing_ms",
    "loss_event",
    "rebuffer_event",
]

TARGET_COLUMN = "congestion"
GROUP_COLUMNS = ["source", "scenario_id", "flow_id"]


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str] = CANONICAL_COLUMNS) -> pd.DataFrame:
    """Return a copy with all canonical columns present."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[list(columns)]


def list_files(root: str | Path, patterns: Iterable[str]) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def read_csv_flexible(path: str | Path, names: list[str] | None = None) -> pd.DataFrame:
    """Read regular, gzipped, or whitespace-separated CSV-like files."""
    path = Path(path)
    try:
        return pd.read_csv(path)
    except Exception:
        if names is not None:
            return pd.read_csv(path, names=names)
        return pd.read_csv(path, sep=None, engine="python")


def write_table(frame: pd.DataFrame, path_without_suffix: str | Path) -> Path:
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = base.with_suffix(".parquet")
        frame.to_parquet(out, index=False)
        return out
    except Exception:
        out = base.with_suffix(".csv")
        frame.to_csv(out, index=False)
        return out


def read_table(path_without_suffix: str | Path) -> pd.DataFrame:
    base = Path(path_without_suffix)
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    if base.exists():
        if base.suffix == ".parquet":
            return pd.read_parquet(base)
        return pd.read_csv(base)
    raise FileNotFoundError(f"No table found for {base}")


def make_flow_id(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = []
    for column in columns:
        if column in frame.columns:
            values.append(frame[column].astype(str))
    if not values:
        return pd.Series(["flow_0"] * len(frame), index=frame.index)
    result = values[0]
    for value in values[1:]:
        result = result + ":" + value
    return result

