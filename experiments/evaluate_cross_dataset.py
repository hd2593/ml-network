from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.schema import read_table
from src.models.evaluate import evaluate_classifier, print_metrics
from src.models.train_tree import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="data/models/random_forest.pkl")
    parser.add_argument("--processed-root", default="data/processed")
    args = parser.parse_args()

    model = load_model(args.model_path)
    root = Path(args.processed_root)
    for name in ["val", "test"]:
        try:
            frame = read_table(root / name)
        except FileNotFoundError:
            continue
        print_metrics(name, evaluate_classifier(model, frame))


if __name__ == "__main__":
    main()
