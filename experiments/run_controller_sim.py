from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.controller.cwnd_policy import apply_cwnd_action, choose_cwnd_action
from src.controller.vanilla_tcp_baseline import cubic_like_baseline
from src.datasets.schema import read_table
from src.models.evaluate import feature_target_split
from src.models.train_tree import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="data/models/random_forest.pkl")
    parser.add_argument("--input", default="data/processed/test")
    parser.add_argument("--output", default="reports/figures/controller_sim.csv")
    args = parser.parse_args()

    model = load_model(args.model_path)
    frame = read_table(args.input).sort_values(["source", "scenario_id", "flow_id", "timestamp"]).head(1000)
    X, _, _ = feature_target_split(frame)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)

    ml_cwnd = 20.0
    vanilla_cwnd = 20.0
    rows = []
    for (_, row), probability in zip(frame.iterrows(), probs):
        action = choose_cwnd_action(float(probability))
        ml_cwnd = apply_cwnd_action(ml_cwnd, action)
        vanilla_cwnd = cubic_like_baseline(vanilla_cwnd, bool(row.get("loss_event", False)))
        rows.append(
            {
                "timestamp": row["timestamp"],
                "probability": float(probability),
                "action": action,
                "ml_cwnd": ml_cwnd,
                "vanilla_cwnd": vanilla_cwnd,
            }
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote controller simulation -> {out}")


if __name__ == "__main__":
    main()

