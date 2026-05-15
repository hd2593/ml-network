# TCP Congestion Prediction

Python pipeline for predicting short-term TCP congestion from real telemetry datasets:

- **Puffer** production TCP / video-streaming telemetry
- **Data Center TCP** clean TCP flow attributes (Zenodo `10894768`)

Two model families are implemented:

1. **Tree baselines** - `RandomForestClassifier`and `XGBoost`.
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
