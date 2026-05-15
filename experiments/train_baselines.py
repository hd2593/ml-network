from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import read_table
from src.features.subsets import restrict_to_features, shared_feature_columns
from src.models.evaluate import evaluate_classifier, print_metrics
from src.models.train_tree import save_model, train_random_forest, train_xgboost


def _try_read(path: Path):
    try:
        return read_table(path)
    except FileNotFoundError:
        import pandas as pd
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["combined", "puffer", "datacenter_tcp"], default="combined")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--model-out", default="data/models/random_forest.pkl")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="Override the number of trees (default: 300).")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Override max tree depth (default: 24 for RF). Use a small value (e.g. 10) for speed.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel jobs. -1 = all cores (default).")
    parser.add_argument("--max-train-rows", type=int, default=None,
                        help="Cap training rows (default: 2,000,000 to keep RAM bounded). Use 0 to disable.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="XGBoost only. RandomForest is CPU-only (sklearn has no GPU backend).")
    parser.add_argument("--feature-subset", choices=["all", "shared"], default="all",
                        help=("'shared' restricts training features to columns populated in "
                              "both Puffer and DataCenter TCP. 'all' uses every numeric feature."))
    parser.add_argument("--metrics-out", default=None,
                        help=("Path to write validation+test metrics JSON. "
                              "Default: reports/figures/{model}_metrics.json."))
    args = parser.parse_args()

    root = Path(args.processed_root)
    train = read_table(root / "train" if args.dataset == "combined" else root / f"{args.dataset}_windows")
    val = read_table(root / "val")
    test = read_table(root / "test")
    print(
        f"Loaded train: {len(train):,} rows | val: {len(val):,} rows | test: {len(test):,} rows",
        flush=True,
    )

    if args.feature_subset == "shared":
        puffer = _try_read(root / "puffer_windows")
        datacenter = _try_read(root / "datacenter_tcp_windows")
        shared = shared_feature_columns([puffer, datacenter])
        if not shared:
            raise SystemExit(
                "Could not compute shared features: at least one of "
                "puffer_windows or datacenter_tcp_windows is missing."
            )
        print(f"[feature-subset=shared] keeping {len(shared)} columns: {shared}", flush=True)
        train = restrict_to_features(train, shared)
        val = restrict_to_features(val, shared)
        test = restrict_to_features(test, shared)

    if args.max_train_rows == 0:
        cap: int | None = None
    elif args.max_train_rows is None:
        cap = 2_000_000
    else:
        cap = args.max_train_rows

    kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "n_jobs": args.n_jobs,
        "max_train_rows": cap,
    }
    if args.model == "random_forest":
        if args.device == "cuda":
            print("[random_forest] note: scikit-learn RandomForest is CPU-only; --device cuda is ignored. "
                  "Use --model xgboost for GPU training.", flush=True)
        model = train_random_forest(train, **kwargs)
    else:
        kwargs["device"] = args.device
        model = train_xgboost(train, **kwargs)
    out = save_model(model, args.model_out)
    print(f"Saved model -> {out}", flush=True)

    val_metrics = evaluate_classifier(model, val)
    print_metrics("validation", val_metrics)
    test_metrics = evaluate_classifier(model, test)
    print_metrics("test", test_metrics)

    metrics_out = args.metrics_out or f"reports/figures/{args.model}_metrics.json"
    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2)
    )
    print(f"Wrote metrics -> {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
