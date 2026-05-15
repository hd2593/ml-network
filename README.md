# TCP Congestion Prediction

Python pipeline for predicting short-term TCP congestion from real telemetry datasets:

- **Puffer** production TCP / video-streaming telemetry
- **Data Center TCP** clean TCP flow attributes (Zenodo `10894768`)

Two model families are implemented:

1. **Tree baselines** - `RandomForestClassifier`, optional XGBoost.
2. **LSTM** - small PyTorch LSTM operating on raw telemetry sequences.

A simple offline `cwnd` controller compares the ML-driven policy against a
Reno-style baseline.

## Quick Start

```powershell
cd ml_tcp_congestion
python -m pip install -r requirements.txt

# Smoke test with conservative limits on real data
python experiments/prepare_datasets.py --datacenter-max-files 30 --puffer-max-rows-per-file 50000
python experiments/train_baselines.py --dataset combined
python experiments/evaluate_cross_dataset.py
python experiments/run_controller_sim.py
python experiments/train_lstm.py --epochs 5
```

Drop the `--*-max-*` flags to use the full datasets once a smoke test passes.
Add `--sample-if-empty` to `prepare_datasets.py` to generate synthetic data
when no external datasets are available.

## Real Dataset Layout

```text
data/external/puffer/
    video_sent_<date_range>.csv
    video_acked_<date_range>.csv          (optional, not currently read)
    client_buffer_<date_range>.csv
data/external/datacenter_tcp/
    <algo>/<algo><run>.log.csv            (e.g. bbr/bbr1.log.csv)
```

The DataCenter TCP archive lives at
`https://zenodo.org/records/10894768/files/tcp-dataset.zip?download=1`.
Puffer daily files come from `https://puffer.stanford.edu/data-description/`.

## Outputs

```text
data/processed/
    puffer_labeled.{parquet,csv}              raw labeled telemetry - used by LSTM
    datacenter_tcp_labeled.{parquet,csv}
    puffer_windows.{parquet,csv}              rolling-window features - used by trees
    datacenter_tcp_windows.{parquet,csv}
    train.{parquet,csv}                       internal training split
    val.{parquet,csv}                         internal validation split
    test.{parquet,csv}                        internal held-out test split
data/models/
    random_forest.pkl
    lstm.pt                                   includes weights, feature names, normalizer
reports/figures/
    controller_sim.csv
    lstm_metrics.json
```

Parquet is used automatically when `pyarrow` is installed; otherwise CSV.

## CLI Knobs For Data Loaders

`prepare_datasets.py` exposes these caps for working with large dumps:

- `--puffer-max-files`, `--puffer-max-rows-per-file`
- `--datacenter-max-files`
