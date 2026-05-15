"""End-to-end LSTM training + evaluation.

The tree pipeline operates on row-level windowed features. The LSTM consumes
*sequences* of raw telemetry over a sliding window, so this script:

1. Loads each per-source labeled telemetry table written by
   ``prepare_datasets.py`` (``puffer_labeled``, ``datacenter_tcp_labeled``).
2. Splits **at the flow level** (so the same flow does not appear in both
   train and val/test) into train/val/test.
3. Materializes sequences using
   :func:`src.features.windowing.build_sequence_tensors`.
4. Trains the small LSTM in :mod:`src.models.train_lstm` and evaluates it on
   validation and held-out test flows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.datasets.schema import GROUP_COLUMNS, read_table
from src.features.splits import (
    apply_manifest,
    load_flow_split_manifest,
    three_way_flow_split,
)
from src.features.windowing import build_sequence_tensors
from src.models.evaluate import print_metrics
from src.models.train_lstm import (
    evaluate_lstm,
    predict_proba,
    save_lstm,
    select_threshold,
    train_lstm,
)


def _load_optional(path_base: Path) -> pd.DataFrame:
    try:
        return read_table(path_base)
    except FileNotFoundError:
        return pd.DataFrame()


def _build_sequences(
    frame: pd.DataFrame,
    window_size: int,
    max_sequences: int | None = None,
    seed: int = 42,
    label: str | None = None,
    with_engineered: bool = False,
    engineered_window: int = 8,
):
    if frame.empty:
        return np.empty((0, window_size, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    return build_sequence_tensors(
        frame,
        window_size=window_size,
        max_sequences=max_sequences,
        random_state=seed,
        label=label,
        with_engineered=with_engineered,
        engineered_window=engineered_window,
    )


def _sequence_cache_key(
    frame: pd.DataFrame,
    window_size: int,
    max_sequences: int | None,
    seed: int,
    with_engineered: bool,
    engineered_window: int,
) -> dict:
    flows = int(frame[GROUP_COLUMNS].drop_duplicates().shape[0]) if not frame.empty else 0
    return {
        "rows": int(len(frame)),
        "flows": flows,
        "window_size": int(window_size),
        "max_sequences": int(max_sequences) if max_sequences else 0,
        "seed": int(seed),
        "with_engineered": bool(with_engineered),
        "engineered_window": int(engineered_window),
    }


def _load_or_build_sequences(
    cache_root: Path | None,
    label: str,
    frame: pd.DataFrame,
    window_size: int,
    max_sequences: int | None,
    seed: int,
    force_rebuild: bool,
    with_engineered: bool,
    engineered_window: int,
):
    if cache_root is None:
        return _build_sequences(
            frame, window_size, max_sequences, seed, label, with_engineered, engineered_window
        )

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    x_path = cache_root / f"{label}_X.npy"
    y_path = cache_root / f"{label}_y.npy"
    feat_path = cache_root / f"{label}_features.json"
    manifest_path = cache_root / f"{label}_manifest.json"
    key = _sequence_cache_key(
        frame, window_size, max_sequences, seed, with_engineered, engineered_window
    )

    if not force_rebuild and all(p.exists() for p in (x_path, y_path, feat_path, manifest_path)):
        try:
            existing = json.loads(manifest_path.read_text())
            if existing == key:
                print(f"[{label}] sequence cache hit -> {x_path}")
                X = np.load(x_path)
                y = np.load(y_path)
                features = json.loads(feat_path.read_text())
                return X, y, features
            print(f"[{label}] sequence cache stale ({existing} != {key}), rebuilding")
        except Exception as exc:
            print(f"[{label}] sequence cache unreadable ({exc}), rebuilding")

    X, y, features = _build_sequences(
        frame, window_size, max_sequences, seed, label, with_engineered, engineered_window
    )
    np.save(x_path, X)
    np.save(y_path, y)
    feat_path.write_text(json.dumps(features))
    manifest_path.write_text(json.dumps(key))
    print(f"[{label}] sequence cache written -> {x_path} ({X.nbytes / 1e9:.2f} GB)")
    return X, y, features


def _align_features(X: np.ndarray, feature_names: list[str], target_names: list[str]) -> np.ndarray:
    if feature_names == target_names:
        return X
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    n, w, _ = X.shape
    aligned = np.zeros((n, w, len(target_names)), dtype=X.dtype)
    for j, name in enumerate(target_names):
        if name in name_to_idx:
            aligned[:, :, j] = X[:, :, name_to_idx[name]]
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the LSTM congestion classifier.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--model-out", default="data/models/lstm.pt")
    parser.add_argument("--metrics-out", default="reports/figures/lstm_metrics.json")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-cache-root", default="data/processed/lstm_sequence_cache",
                        help="Directory for cached materialized sequences. Empty string disables caching.")
    parser.add_argument("--rebuild-sequences", action="store_true",
                        help="Ignore cache and rematerialize sequences.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience on val_loss. Set 0 to disable.")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global gradient norm. 0 disables clipping.")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-engineered-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Augment each timestep with causal rolling mean/std/min/max/trend.")
    parser.add_argument("--engineered-window", type=int, default=8,
                        help="Causal lookback window for per-step engineered stats.")
    parser.add_argument("--class-weighted", action=argparse.BooleanOptionalAction, default=True,
                        help="Use inverse-frequency class weights in CrossEntropyLoss.")
    parser.add_argument("--threshold-objective",
                        choices=["f1", "accuracy", "balanced_accuracy", "fixed"], default="f1",
                        help="Metric used to pick the decision threshold on validation predictions.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold when --threshold-objective=fixed.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-sequences", type=int, default=2_000_000,
                        help="Cap train sequences during materialization. Use 0 to keep every sequence.")
    parser.add_argument("--max-eval-sequences", type=int, default=0,
                        help="Cap validation/test sequences during materialization. Use 0 to keep every sequence.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Force a specific torch device. 'auto' picks CUDA when available.")
    args = parser.parse_args()

    root = Path(args.processed_root)

    print("Loading labeled telemetry...")
    puffer = _load_optional(root / "puffer_labeled")
    datacenter = _load_optional(root / "datacenter_tcp_labeled")
    for name, frame in [("puffer", puffer), ("datacenter_tcp", datacenter)]:
        print(f"  {name}: {len(frame)} rows, {frame[GROUP_COLUMNS].drop_duplicates().shape[0] if not frame.empty else 0} flows")

    train_sources = [df for df in [puffer, datacenter] if not df.empty]
    if not train_sources:
        raise SystemExit(
            "No labeled telemetry found. Run experiments/prepare_datasets.py first "
            "so that data/processed/{puffer,datacenter_tcp}_labeled.{parquet,csv} exist."
        )
    combined_train = pd.concat(train_sources, ignore_index=True)
    print(f"Combined Puffer+DataCenter telemetry: {len(combined_train)} rows")

    manifest_path = root / "flow_splits.json"
    if manifest_path.exists():
        print(f"Loading flow-split manifest -> {manifest_path}")
        manifest = load_flow_split_manifest(manifest_path)
        train_frame = apply_manifest(combined_train, manifest, "train")
        val_frame = apply_manifest(combined_train, manifest, "val")
        test_frame = apply_manifest(combined_train, manifest, "test")
    else:
        print("flow_splits.json missing; computing fresh flow-level split")
        train_frame, val_frame, test_frame = three_way_flow_split(
            combined_train,
            val_size=args.val_fraction,
            test_size=0.15,
            seed=args.seed,
        )

    print(f"Train flows: {train_frame[GROUP_COLUMNS].drop_duplicates().shape[0]}")
    print(f"Val flows:   {val_frame[GROUP_COLUMNS].drop_duplicates().shape[0]}")
    print(f"Test flows:  {test_frame[GROUP_COLUMNS].drop_duplicates().shape[0]}")

    max_train_sequences = args.max_train_sequences if args.max_train_sequences and args.max_train_sequences > 0 else None
    max_eval_sequences = args.max_eval_sequences if args.max_eval_sequences and args.max_eval_sequences > 0 else None
    cache_root = Path(args.sequence_cache_root) if args.sequence_cache_root else None

    print("Materializing sequences...")
    X_train, y_train, feature_names = _load_or_build_sequences(
        cache_root, "train", train_frame, args.window_size,
        max_train_sequences, args.seed, args.rebuild_sequences,
        args.use_engineered_features, args.engineered_window,
    )
    X_val, y_val, val_features = _load_or_build_sequences(
        cache_root, "val", val_frame, args.window_size,
        max_eval_sequences, args.seed + 10, args.rebuild_sequences,
        args.use_engineered_features, args.engineered_window,
    )
    X_test, y_test, test_features = _load_or_build_sequences(
        cache_root, "test", test_frame, args.window_size,
        max_eval_sequences, args.seed + 20, args.rebuild_sequences,
        args.use_engineered_features, args.engineered_window,
    )
    X_val = _align_features(X_val, val_features, feature_names)
    X_test = _align_features(X_test, test_features, feature_names)
    print(f"  train: {X_train.shape}, positive rate: {float(np.mean(y_train)) if len(y_train) else 0.0:.3f}")
    print(f"  val:   {X_val.shape}")
    print(f"  test:  {X_test.shape}")

    if len(X_train) == 0:
        raise SystemExit("No training sequences available — labeled telemetry is too short for the chosen window size.")

    print("Training...")
    model, artifact, mean, std, history = train_lstm(
        X_train,
        y_train,
        X_val=X_val if len(X_val) else None,
        y_val=y_val if len(y_val) else None,
        feature_names=feature_names,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weighted=args.class_weighted,
        device=args.device,
        patience=args.patience if args.patience > 0 else None,
        bidirectional=args.bidirectional,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    save_path = save_lstm(artifact, mean, std, args.model_out)
    print(f"Saved LSTM -> {save_path}")

    if args.threshold_objective == "fixed" or not len(X_val):
        chosen_threshold = float(args.threshold)
    else:
        val_prob = predict_proba(model, X_val, mean, std)
        chosen_threshold = select_threshold(val_prob, y_val, args.threshold_objective)
        print(
            f"Selected decision threshold {chosen_threshold:.3f} "
            f"(objective={args.threshold_objective}) on validation"
        )

    results = {
        "history": history,
        "threshold": chosen_threshold,
        "threshold_objective": args.threshold_objective,
        "class_weighted": bool(args.class_weighted),
        "use_engineered_features": bool(args.use_engineered_features),
        "engineered_window": int(args.engineered_window),
        "bidirectional": bool(args.bidirectional),
    }
    if len(X_val):
        val_metrics = evaluate_lstm(model, X_val, y_val, mean, std, threshold=chosen_threshold)
        print_metrics("validation", val_metrics)
        results["validation"] = val_metrics
        val_metrics_default = evaluate_lstm(model, X_val, y_val, mean, std, threshold=0.5)
        results["validation_at_0.5"] = val_metrics_default
    if len(X_test):
        test_metrics = evaluate_lstm(model, X_test, y_test, mean, std, threshold=chosen_threshold)
        print_metrics("test", test_metrics)
        results["test"] = test_metrics
        test_metrics_default = evaluate_lstm(model, X_test, y_test, mean, std, threshold=0.5)
        results["test_at_0.5"] = test_metrics_default

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote LSTM metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
