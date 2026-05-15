from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ensure_columns, list_files


ALIASES = {
    "rtt_ms": ["rtt", "tcpi_rtt", "rtt_ms"],
    "min_rtt_ms": ["min_rtt", "tcpi_min_rtt", "min_rtt_ms"],
    "delivery_rate_bps": ["delivery_rate", "tcpi_delivery_rate", "rate", "bandwidth"],
    "cwnd": ["cwnd", "snd_cwnd", "tcpi_snd_cwnd"],
    "bytes_sent": ["bytes_sent", "sent_bytes"],
    "bytes_acked": ["bytes_acked", "acked_bytes"],
    "bytes_retrans": ["bytes_retrans", "retrans", "retrans_bytes"],
    "segments_out": ["segs_out", "segments_out", "data_segs_out"],
    "segments_in": ["segs_in", "segments_in"],
}


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _numeric_alias(frame: pd.DataFrame, canonical: str) -> pd.Series:
    column = _find_column(frame, ALIASES.get(canonical, [canonical]))
    if column is None:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _read_dc_csv(file: Path) -> pd.DataFrame:
    """Read a Data Center TCP log.

    The Zenodo dataset files are semicolon-delimited and the first column
    (``wscale``) contains values like ``7,7`` that confuse the default comma
    parser. We try semicolon first, then fall back.
    """
    for sep in (";", ",", None):
        try:
            kwargs = {"sep": sep}
            if sep is None:
                kwargs["engine"] = "python"
            frame = pd.read_csv(file, **kwargs)
        except Exception:
            continue
        if frame.shape[1] > 1:
            return frame.dropna(axis=1, how="all")
    return pd.DataFrame()


def load_datacenter_tcp(root: str | Path, max_files: int | None = None) -> pd.DataFrame:
    """Load Data Center TCP CSV files into canonical telemetry rows."""
    root = Path(root)
    files = list_files(root, ["*.csv", "*.txt"])
    if max_files is not None:
        files = files[:max_files]
    print(f"[datacenter_tcp] reading {len(files)} log files from {root}", flush=True)
    frames = []

    for file_idx, file in enumerate(files, start=1):
        if file_idx == 1 or file_idx % 50 == 0 or file_idx == len(files):
            print(f"[datacenter_tcp] ({file_idx}/{len(files)}) {file.parent.name}/{file.name}", flush=True)
        raw = _read_dc_csv(file)
        if raw.empty:
            continue
        scenario = file.parent.name if file.parent != root else file.stem
        flow_id = f"{scenario}:{file.stem}"
        time_col = _find_column(raw, ["timestamp", "time", "elapsed", "lastrcv"])
        if time_col is not None:
            timestamp = pd.to_numeric(raw[time_col], errors="coerce") / 1000.0
        else:
            timestamp = pd.Series(np.arange(len(raw), dtype=float))

        frame = pd.DataFrame(
            {
                "source": "datacenter_tcp",
                "scenario_id": scenario,
                "flow_id": flow_id,
                "timestamp": timestamp.to_numpy(),
                "rtt_ms": _numeric_alias(raw, "rtt_ms").to_numpy(),
                "min_rtt_ms": _numeric_alias(raw, "min_rtt_ms").to_numpy(),
                "delivery_rate_bps": _numeric_alias(raw, "delivery_rate_bps").to_numpy(),
                "cwnd": _numeric_alias(raw, "cwnd").to_numpy(),
                "bytes_sent": _numeric_alias(raw, "bytes_sent").to_numpy(),
                "bytes_acked": _numeric_alias(raw, "bytes_acked").to_numpy(),
                "bytes_retrans": _numeric_alias(raw, "bytes_retrans").to_numpy(),
                "segments_out": _numeric_alias(raw, "segments_out").to_numpy(),
                "segments_in": _numeric_alias(raw, "segments_in").to_numpy(),
            }
        )
        frame["throughput_bps"] = frame.groupby("flow_id")["bytes_acked"].diff() * 8.0 / frame.groupby("flow_id")[
            "timestamp"
        ].diff().replace(0, np.nan)
        frame["loss_event"] = frame.groupby("flow_id")["bytes_retrans"].diff().fillna(0) > 0
        frames.append(ensure_columns(frame))

    if not frames:
        return ensure_columns(pd.DataFrame())
    return pd.concat(frames, ignore_index=True)

