from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.load_datacenter_tcp import load_datacenter_tcp
from src.datasets.load_puffer import load_puffer
from src.datasets.schema import write_table
from src.features.labels import add_congestion_label
from src.features.splits import three_way_flow_split, write_flow_split_manifest
from src.features.windowing import build_sliding_window_features


def make_sample_telemetry(n: int = 500, source: str = "sample") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for scenario in range(3):
        rtt = 40 + scenario * 15 + rng.normal(0, 2, n)
        delivery = 8e6 + rng.normal(0, 5e5, n)
        cwnd = 30 + rng.normal(0, 3, n)
        congestion_points = rng.choice(np.arange(40, n - 20), size=5, replace=False)
        for point in congestion_points:
            rtt[point : point + 10] += np.linspace(5, 45, 10)
            delivery[point : point + 10] *= np.linspace(1.0, 0.45, 10)
            cwnd[point : point + 10] *= np.linspace(1.0, 0.7, 10)
        for i in range(n):
            rows.append(
                {
                    "source": source,
                    "scenario_id": f"scenario_{scenario}",
                    "flow_id": f"flow_{scenario}",
                    "timestamp": float(i),
                    "rtt_ms": max(1.0, rtt[i]),
                    "min_rtt_ms": 30 + scenario * 10,
                    "delivery_rate_bps": max(1.0, delivery[i]),
                    "throughput_bps": max(1.0, delivery[i] * rng.uniform(0.8, 1.05)),
                    "cwnd": max(1.0, cwnd[i]),
                    "in_flight": max(1.0, cwnd[i] * rng.uniform(0.4, 1.1)),
                    "ack_spacing_ms": rng.uniform(1, 20),
                    "loss_event": bool(rng.random() < 0.02),
                    "rebuffer_event": False,
                }
            )
    return pd.DataFrame(rows)


def load_all_raw(
    data_root: str | Path,
    sample_if_empty: bool = False,
    puffer_kwargs: dict | None = None,
    datacenter_kwargs: dict | None = None,
) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    puffer_kwargs = puffer_kwargs or {}
    datacenter_kwargs = datacenter_kwargs or {}
    datasets = {
        "puffer": load_puffer(data_root / "external" / "puffer", **puffer_kwargs),
        "datacenter_tcp": load_datacenter_tcp(data_root / "external" / "datacenter_tcp", **datacenter_kwargs),
    }
    if sample_if_empty and all(frame.empty for frame in datasets.values()):
        datasets["puffer"] = make_sample_telemetry(source="puffer")
        datasets["datacenter_tcp"] = make_sample_telemetry(source="datacenter_tcp")
    return datasets


def prepare_datasets(
    data_root: str | Path = "data",
    output_root: str | Path = "data/processed",
    sample_if_empty: bool = False,
    puffer_kwargs: dict | None = None,
    datacenter_kwargs: dict | None = None,
):
    output_root = Path(output_root)
    raw = load_all_raw(
        data_root,
        sample_if_empty=sample_if_empty,
        puffer_kwargs=puffer_kwargs,
        datacenter_kwargs=datacenter_kwargs,
    )
    windowed = {}
    for name, frame in raw.items():
        print(f"[{name}] raw rows: {len(frame):,}", flush=True)
        labeled = add_congestion_label(frame, label=name)
        if not labeled.empty:
            raw_path = write_table(labeled, output_root / f"{name}_labeled")
            print(f"[{name}] wrote labeled telemetry: {len(labeled):,} rows -> {raw_path}", flush=True)
        features = build_sliding_window_features(labeled, label=name)
        if not features.empty:
            path = write_table(features, output_root / f"{name}_windows")
            print(f"[{name}] wrote windowed features: {len(features):,} rows -> {path}", flush=True)
        windowed[name] = features

    train_sources = [windowed[name] for name in ["puffer", "datacenter_tcp"] if not windowed[name].empty]
    if not train_sources:
        raise RuntimeError("No training data found. Add datasets or run with --sample-if-empty.")
    train_frame = pd.concat(train_sources, ignore_index=True)

    train, val, test_frame = three_way_flow_split(
        train_frame, val_size=0.15, test_size=0.15, seed=42
    )
    write_flow_split_manifest(
        train,
        val,
        test_frame,
        output_root / "flow_splits.json",
        seed=42,
        val_size=0.15,
        test_size=0.15,
    )

    paths = {
        "train": write_table(train, output_root / "train"),
        "val": write_table(val, output_root / "val"),
        "test": write_table(test_frame, output_root / "test"),
    }
    for split, path in paths.items():
        print(f"Wrote {split}: {path}")
    return paths
