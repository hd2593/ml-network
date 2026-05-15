from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_dataset import prepare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build labelled, windowed feature tables from Puffer and DataCenter TCP. "
            "Use the --max-* flags to control memory/time on large datasets."
        )
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--sample-if-empty", action="store_true",
                        help="Generate synthetic data when no external datasets are present.")

    parser.add_argument("--puffer-max-files", type=int, default=None,
                        help="Cap the number of video_sent CSVs read.")
    parser.add_argument("--puffer-max-rows-per-file", type=int, default=None,
                        help="Optional row cap inside each Puffer file (for smoke tests).")

    parser.add_argument("--datacenter-max-files", type=int, default=None,
                        help="Cap the number of DataCenter TCP logs read.")

    args = parser.parse_args()

    puffer_kwargs = {
        "max_files": args.puffer_max_files,
        "max_rows_per_file": args.puffer_max_rows_per_file,
    }
    datacenter_kwargs = {"max_files": args.datacenter_max_files}

    prepare_datasets(
        args.data_root,
        args.output_root,
        sample_if_empty=args.sample_if_empty,
        puffer_kwargs=puffer_kwargs,
        datacenter_kwargs=datacenter_kwargs,
    )


if __name__ == "__main__":
    main()
