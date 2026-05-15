from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_probability_timeline(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["timestamp"], frame["congestion_probability"], label="congestion probability")
    if "congestion" in frame.columns:
        ax.scatter(frame["timestamp"], frame["congestion"], s=8, alpha=0.4, label="label")
    ax.axhline(0.65, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

