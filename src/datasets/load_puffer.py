from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ensure_columns, list_files, make_flow_id


VIDEO_SENT_COLUMN_ALIASES = {
    "time": ["time", "time (ns GMT)", "time_ns"],
    "session_id": ["session_id"],
    "index": ["index"],
    "expt_id": ["expt_id"],
    "channel": ["channel"],
    "size": ["size"],
    "cwnd": ["cwnd"],
    "in_flight": ["in_flight"],
    "min_rtt": ["min_rtt"],
    "rtt": ["rtt"],
    "delivery_rate": ["delivery_rate"],
}

CLIENT_BUFFER_COLUMN_ALIASES = {
    "time": ["time", "time (ns GMT)", "time_ns"],
    "session_id": ["session_id"],
    "index": ["index"],
    "event": ["event"],
}


def _normalize_columns(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename source columns to canonical names so downstream logic can rely on them."""
    rename = {}
    columns_lower = {str(column).lower(): column for column in frame.columns}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            actual = columns_lower.get(candidate.lower())
            if actual is not None and actual != canonical:
                rename[actual] = canonical
                break
    if rename:
        frame = frame.rename(columns=rename)
    return frame


def _read_puffer_csv(path: Path, aliases: dict[str, list[str]], usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a Puffer CSV and rename headers like ``time (ns GMT)`` to ``time``."""
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return pd.DataFrame()
    head = _normalize_columns(head, aliases)
    actual_cols = list(head.columns)
    keep = None
    if usecols is not None:
        keep = [column for column in usecols if column in actual_cols]
    rename_map: dict[str, str] = {}
    columns_lower = {str(column).lower(): column for column in actual_cols}
    for canonical, candidates in aliases.items():
        if canonical in actual_cols:
            continue
        for candidate in candidates:
            real = columns_lower.get(candidate.lower())
            if real is not None:
                rename_map[real] = canonical
                break
    read_kwargs = {}
    if keep is not None:
        read_kwargs["usecols"] = keep
    frame = pd.read_csv(path, **read_kwargs)
    frame = _normalize_columns(frame, aliases)
    return frame


def _iter_rebuffer_events(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Stream client_buffer files in chunks and keep only rebuffer events.

    These files routinely exceed 2 GB so reading them in one shot can OOM.
    Filtering inside the chunk loop keeps memory bounded.
    """
    pieces = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk = _normalize_columns(chunk, CLIENT_BUFFER_COLUMN_ALIASES)
        if "event" not in chunk.columns:
            continue
        keep = chunk[chunk["event"].astype(str).str.lower().eq("rebuffer")]
        if not keep.empty:
            pieces.append(keep[["time", "session_id", "index"]].copy())
    if not pieces:
        return pd.DataFrame(columns=["time", "session_id", "index"])
    return pd.concat(pieces, ignore_index=True)


def _load_rebuffer_events(root: Path) -> pd.DataFrame:
    files = list_files(root, ["client_buffer*.csv", "client_buffer*.csv.gz"])
    frames = []
    for file in files:
        events = _iter_rebuffer_events(file)
        if events.empty:
            continue
        events["flow_id"] = make_flow_id(events, ["session_id", "index"])
        events["timestamp"] = pd.to_numeric(events["time"], errors="coerce") / 1e9
        frames.append(events[["flow_id", "timestamp"]].dropna())
    if not frames:
        return pd.DataFrame(columns=["flow_id", "timestamp"])
    return pd.concat(frames, ignore_index=True)


def _mark_future_rebuffer(samples: pd.DataFrame, events: pd.DataFrame, horizon_seconds: float) -> pd.Series:
    if events.empty or samples.empty:
        return pd.Series(False, index=samples.index)
    result = pd.Series(False, index=samples.index)
    event_groups = {flow: group["timestamp"].sort_values().to_numpy() for flow, group in events.groupby("flow_id")}
    for flow, idx in samples.groupby("flow_id").groups.items():
        times = samples.loc[idx, "timestamp"].to_numpy(dtype=float)
        event_times = event_groups.get(flow)
        if event_times is None or len(event_times) == 0:
            continue
        pos = np.searchsorted(event_times, times, side="left")
        has_future = pos < len(event_times)
        next_event = np.full(len(times), np.inf)
        next_event[has_future] = event_times[pos[has_future]]
        result.loc[idx] = (next_event >= times) & (next_event <= times + horizon_seconds)
    return result


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_puffer(
    root: str | Path,
    horizon_seconds: float = 4.0,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
) -> pd.DataFrame:
    """Load Puffer CSVs into canonical telemetry rows.

    Args:
        root: directory containing ``video_sent*.csv`` and ``client_buffer*.csv``.
        horizon_seconds: how far ahead a rebuffer counts as ``rebuffer_event``.
        max_files: optional cap on number of ``video_sent`` files processed.
        max_rows_per_file: optional row cap per file (for quick smoke tests).
    """
    root = Path(root)
    files = list_files(root, ["video_sent*.csv", "video_sent*.csv.gz"])
    if max_files is not None:
        files = files[:max_files]
    _log(f"[puffer] reading {len(files)} video_sent files from {root}")
    frames = []
    _log("[puffer] streaming client_buffer files for rebuffer events...")
    rebuffer_events = _load_rebuffer_events(root)
    _log(f"[puffer] collected {len(rebuffer_events):,} rebuffer events")

    for file_idx, file in enumerate(files, start=1):
        _log(f"[puffer] ({file_idx}/{len(files)}) reading {file.name}")
        raw = _read_puffer_csv(file, VIDEO_SENT_COLUMN_ALIASES)
        if raw.empty:
            continue
        if max_rows_per_file is not None and len(raw) > max_rows_per_file:
            raw = raw.head(max_rows_per_file)
        _log(f"[puffer] ({file_idx}/{len(files)}) rows={len(raw):,}")
        raw["flow_id"] = make_flow_id(raw, ["session_id", "index"])
        raw["timestamp"] = pd.to_numeric(raw["time"], errors="coerce") / 1e9
        raw = raw.sort_values(["flow_id", "timestamp"])

        timestamp_delta = raw.groupby("flow_id")["timestamp"].diff()
        size_bits = pd.to_numeric(raw.get("size"), errors="coerce") * 8.0
        throughput = size_bits / timestamp_delta.replace(0, np.nan)

        frame = pd.DataFrame(
            {
                "source": "puffer",
                "scenario_id": raw.get("expt_id", pd.Series("unknown", index=raw.index)).astype(str).to_numpy(),
                "flow_id": raw["flow_id"].to_numpy(),
                "timestamp": raw["timestamp"].to_numpy(),
                "rtt_ms": (pd.to_numeric(raw.get("rtt"), errors="coerce") / 1000.0).to_numpy(),
                "min_rtt_ms": (pd.to_numeric(raw.get("min_rtt"), errors="coerce") / 1000.0).to_numpy(),
                "delivery_rate_bps": (pd.to_numeric(raw.get("delivery_rate"), errors="coerce") * 8.0).to_numpy(),
                "throughput_bps": throughput.to_numpy(),
                "cwnd": pd.to_numeric(raw.get("cwnd"), errors="coerce").to_numpy(),
                "in_flight": pd.to_numeric(raw.get("in_flight"), errors="coerce").to_numpy(),
                "ack_spacing_ms": (raw.groupby("flow_id")["timestamp"].diff() * 1000.0).to_numpy(),
                "packet_size_bytes": pd.to_numeric(raw.get("size"), errors="coerce").to_numpy(),
            }
        )
        frame["rebuffer_event"] = _mark_future_rebuffer(frame, rebuffer_events, horizon_seconds)
        frames.append(ensure_columns(frame))

    if not frames:
        return ensure_columns(pd.DataFrame())
    return pd.concat(frames, ignore_index=True)
