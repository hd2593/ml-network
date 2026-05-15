"""Small LSTM congestion classifier and helpers.

Architecture (matches PLAN.md):

    LSTM(hidden=32, layers=1, dropout=0.1) -> Linear(32, 16) -> ReLU -> Linear(16, 2)

The model consumes sequences of shape ``[batch, window_size, n_features]`` and
emits class logits for ``no_congestion`` (0) vs ``congestion`` (1).

The training code only imports ``torch`` lazily so the rest of the project can
run without it installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is optional. Install torch to use the LSTM.") from exc
    return torch, nn, DataLoader, TensorDataset


def resolve_device(preferred: str | None = None):
    """Pick a torch device. ``preferred`` may be ``'cuda'``, ``'cpu'``, ``'auto'`` or None."""
    torch, _, _, _ = _import_torch()
    if preferred in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Verify that the active interpreter has a CUDA-enabled torch wheel "
            "and that NVIDIA drivers are installed."
        )
    return torch.device(preferred)


def describe_device(device) -> str:
    torch, _, _, _ = _import_torch()
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        free, total = torch.cuda.mem_get_info(idx)
        return f"cuda:{idx} ({name}, {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total)"
    return "cpu"


def build_lstm(
    feature_count: int,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    bidirectional: bool = False,
):
    torch, nn, _, _ = _import_torch()

    directions = 2 if bidirectional else 1
    head_in = hidden_size * directions
    head_hidden = max(32, head_in // 4)

    class CongestionLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                feature_count,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(head_in, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 2),
            )

        def forward(self, batch):
            _, (hidden, _) = self.lstm(batch)
            if self.bidirectional:
                last = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                last = hidden[-1]
            return self.head(last)

    return CongestionLSTM()


@dataclass
class LSTMArtifact:
    """Container with everything needed to reload an LSTM for inference."""
    state_dict: dict
    feature_names: list[str]
    window_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool = False


def _per_feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.subtract(X, mean, dtype=np.float32)
    out /= std
    return out


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    class_weighted: bool = True,
    verbose: bool = True,
    device: str | None = None,
    patience: int | None = 5,
    bidirectional: bool = False,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    lr_min: float = 1e-5,
):
    """Train an LSTM classifier with AdamW + ReduceLROnPlateau + grad clip.

    Returns a tuple ``(model, artifact, mean, std, history)``.

    When ``patience`` is a positive int and a validation set is supplied, the
    best val_loss checkpoint is retained and training stops after ``patience``
    epochs without improvement. The returned ``model`` and ``artifact`` hold
    the best weights, not the last ones.
    """
    torch, nn, DataLoader, TensorDataset = _import_torch()
    torch.backends.cudnn.benchmark = True

    feature_count = X_train.shape[-1]
    window_size = X_train.shape[1]
    mean, std = _per_feature_stats(X_train)
    X_train_norm = _standardize(X_train, mean, std)

    dev = resolve_device(device)
    if verbose:
        print(f"Using device: {describe_device(dev)}")
    model = build_lstm(
        feature_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=lr_min
    )
    if class_weighted:
        positive = float(np.sum(y_train == 1))
        negative = float(np.sum(y_train == 0))
        total = max(1.0, positive + negative)
        weight_neg = total / (2.0 * max(1.0, negative))
        weight_pos = total / (2.0 * max(1.0, positive))
        weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=dev)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def _put_tensors(X_np: np.ndarray, y_np: np.ndarray, name: str):
        """Build (X, y) tensors and put them on ``dev`` if they fit; else keep pinned on host."""
        X_t = torch.from_numpy(np.ascontiguousarray(X_np, dtype=np.float32))
        y_t = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.int64))
        on_gpu = False
        if dev.type == "cuda":
            try:
                X_t = X_t.to(dev)
                y_t = y_t.to(dev)
                on_gpu = True
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
                torch.cuda.empty_cache()
                if verbose:
                    print(f"  {name}: GPU resident failed ({exc}); falling back to pinned host tensors")
                try:
                    X_t = X_t.pin_memory()
                    y_t = y_t.pin_memory()
                except RuntimeError:
                    pass
        if verbose:
            bytes_gb = X_t.element_size() * X_t.nelement() / 1e9
            loc = "GPU" if on_gpu else ("pinned host" if dev.type == "cuda" else "host")
            print(f"  {name}: {tuple(X_t.shape)} on {loc} ({bytes_gb:.2f} GB)")
        return X_t, y_t, on_gpu

    X_train_t, y_train_t, train_on_gpu = _put_tensors(X_train_norm, y_train, "train")
    del X_train_norm
    n_train = X_train_t.shape[0]

    if X_val is not None and y_val is not None and len(X_val):
        X_val_norm = _standardize(X_val, mean, std)
        X_val_t, y_val_t, val_on_gpu = _put_tensors(X_val_norm, y_val, "val")
        del X_val_norm
        n_val = X_val_t.shape[0]
    else:
        X_val_t = y_val_t = None
        val_on_gpu = False
        n_val = 0

    early_stop_active = patience is not None and patience > 0 and X_val_t is not None
    best_val: float | None = None
    best_state: dict | None = None
    best_epoch: int | None = None
    epochs_since_improvement = 0

    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=X_train_t.device)
        loss_sum = torch.zeros((), device=dev)
        batch_count = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            if not train_on_gpu and dev.type == "cuda":
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += loss.detach()
            batch_count += 1
        avg_train = float(loss_sum.item()) / max(1, batch_count)

        avg_val = None
        if X_val_t is not None:
            model.eval()
            val_loss_sum = torch.zeros((), device=dev)
            val_batch_count = 0
            with torch.no_grad():
                for start in range(0, n_val, batch_size):
                    xb = X_val_t[start : start + batch_size]
                    yb = y_val_t[start : start + batch_size]
                    if not val_on_gpu and dev.type == "cuda":
                        xb = xb.to(dev, non_blocking=True)
                        yb = yb.to(dev, non_blocking=True)
                    val_loss_sum += loss_fn(model(xb), yb)
                    val_batch_count += 1
            avg_val = float(val_loss_sum.item()) / max(1, val_batch_count)
            scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val, "lr": current_lr}
        )
        if verbose:
            msg = f"epoch={epoch + 1:>2} train_loss={avg_train:.4f}"
            if avg_val is not None:
                msg += f" val_loss={avg_val:.4f}"
            msg += f" lr={current_lr:.2e}"
            print(msg)

        if early_stop_active and avg_val is not None:
            if best_val is None or avg_val < best_val:
                best_val = avg_val
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    if verbose:
                        print(
                            f"Early stop: no val_loss improvement for {patience} epochs "
                            f"(best val_loss={best_val:.4f} at epoch {best_epoch})."
                        )
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        if verbose:
            print(f"Restored best weights from epoch {best_epoch} (val_loss={best_val:.4f}).")

    artifact = LSTMArtifact(
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        feature_names=list(feature_names) if feature_names is not None else [],
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return model, artifact, mean, std, history


def save_lstm(artifact: LSTMArtifact, mean: np.ndarray, std: np.ndarray, path: str | Path) -> Path:
    """Persist model weights + metadata to ``path``."""
    torch, _, _, _ = _import_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": artifact.state_dict,
            "feature_names": artifact.feature_names,
            "window_size": artifact.window_size,
            "hidden_size": artifact.hidden_size,
            "num_layers": artifact.num_layers,
            "dropout": artifact.dropout,
            "bidirectional": artifact.bidirectional,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        path,
    )
    return path


def load_lstm(path: str | Path, device: str | None = None):
    torch, _, _, _ = _import_torch()
    dev = resolve_device(device)
    blob = torch.load(path, map_location=dev, weights_only=False)
    feature_count = len(blob["feature_names"]) or blob["state_dict"]["lstm.weight_ih_l0"].shape[1]
    model = build_lstm(
        feature_count,
        hidden_size=blob["hidden_size"],
        num_layers=blob["num_layers"],
        dropout=blob["dropout"],
        bidirectional=blob.get("bidirectional", False),
    ).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    mean = np.asarray(blob["mean"], dtype=np.float32)
    std = np.asarray(blob["std"], dtype=np.float32)
    return model, blob["feature_names"], blob["window_size"], mean, std


def predict_proba(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    torch, _, _, _ = _import_torch()
    device = next(model.parameters()).device
    X_norm = _standardize(X, mean, std)
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_norm), batch_size):
            batch = torch.from_numpy(X_norm[start : start + batch_size]).to(device, non_blocking=(device.type == "cuda"))
            logits = model(batch)
            soft = torch.softmax(logits, dim=-1)
            probs.append(soft[:, 1].cpu().numpy())
    if not probs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def select_threshold(
    prob: np.ndarray,
    y: np.ndarray,
    objective: str = "f1",
) -> float:
    """Choose the decision threshold that maximizes ``objective`` on (prob, y).

    Sweeps ``np.linspace(0.05, 0.95, 181)``. Tie-break prefers the threshold
    closest to 0.5 (most calibrated). Returns 0.5 if y is degenerate.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    scorer = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }.get(objective)
    if scorer is None:
        raise ValueError(
            f"Unknown threshold objective '{objective}'. "
            "Expected one of 'f1', 'accuracy', 'balanced_accuracy'."
        )
    thresholds = np.linspace(0.05, 0.95, 181)
    best_score = -np.inf
    best_thr = 0.5
    for thr in thresholds:
        score = scorer(y, (prob >= thr).astype(int))
        # Strictly better -> take; equal-score tie-break -> closer to 0.5
        if score > best_score or (
            np.isclose(score, best_score) and abs(thr - 0.5) < abs(best_thr - 0.5)
        ):
            best_score = score
            best_thr = float(thr)
    return best_thr


def evaluate_lstm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prob = predict_proba(model, X, mean, std)
    pred = (prob >= threshold).astype(int)
    metrics = {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    metrics["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None
    return metrics


def train_lstm_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str | Path,
    epochs: int = 10,
    feature_names: list[str] | None = None,
):
    """Backwards-compatible thin wrapper around :func:`train_lstm`."""
    model, artifact, mean, std, _history = train_lstm(
        X,
        y,
        epochs=epochs,
        feature_names=feature_names,
    )
    return save_lstm(artifact, mean, std, output_path)
