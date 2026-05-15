from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
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
]


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def add_derived_tcp_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in BASE_FEATURE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    in_flight = _series(result, "in_flight")
    cwnd = _series(result, "cwnd")
    rtt = _series(result, "rtt_ms")
    min_rtt = _series(result, "min_rtt_ms")
    bytes_retrans = _series(result, "bytes_retrans")
    bytes_sent = _series(result, "bytes_sent")
    segments_in = _series(result, "segments_in")
    segments_out = _series(result, "segments_out")

    result["inflight_cwnd_ratio"] = in_flight / cwnd.replace(0, np.nan)
    result["rtt_queue_ms"] = rtt - min_rtt
    result["retrans_rate"] = bytes_retrans.diff().clip(lower=0) / bytes_sent.diff().replace(0, np.nan)
    result["ack_to_data_ratio"] = segments_in.diff() / segments_out.diff().replace(0, np.nan)
    return result


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"congestion", "timestamp"}
    return [
        column
        for column in frame.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(frame[column])
    ]
