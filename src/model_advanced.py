"""Project 11: NASA C-MAPSS Turbofan RUL - Advanced model.

Sequence model on raw sensor windows. Two architectures provided:
    - LSTMRegressor      (default): 2-layer LSTM + linear head.
    - TransformerRegressor      : 1D-Transformer (positional encoding + self-attention).

Switch via the `MODEL` environment variable: `MODEL=transformer python src/model_advanced.py`

Pipeline
--------
1. Load FD001 train / test / RUL.
2. Drop low-variance sensors, z-score on train statistics.
3. Build piece-wise linear RUL labels with cap = 125.
4. Build sliding windows of length 30 (raw sensor sequence per sample).
5. Hold out 20% of engines as validation.
6. Train (Adam, MSE), early-stop on val RMSE.
7. Score on the NASA test set (last 30 cycles per engine).
8. Save model + metrics to deliverables/.

Run
---
    python src/model_advanced.py                # LSTM
    MODEL=transformer python src/model_advanced.py

Author: Sandeep Grover, Independent Research.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


HERE         = Path(__file__).resolve().parent
PROJECT_DIR  = HERE.parent
DATA_DIR     = PROJECT_DIR / "data"
CMAPS_DIR    = DATA_DIR / "CMaps" if (DATA_DIR / "CMaps").exists() else DATA_DIR
DELIVERABLES = PROJECT_DIR / "deliverables"
DELIVERABLES.mkdir(parents=True, exist_ok=True)

SUBSET       = "FD001"
WINDOW       = 30
RUL_CAP      = 125
BATCH_SIZE   = 256
EPOCHS       = 60
PATIENCE     = 10
LR           = 1e-3
HIDDEN       = 64
DROPOUT      = 0.2
RANDOM_SEED  = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_subset(subset: str = SUBSET) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(CMAPS_DIR / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    test  = pd.read_csv(CMAPS_DIR / f"test_{subset}.txt",  sep=r"\s+", header=None, names=COLS)
    rul   = pd.read_csv(CMAPS_DIR / f"RUL_{subset}.txt",   sep=r"\s+", header=None, names=["rul"])
    return train, test, rul


def add_piecewise_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["rul"] = np.minimum(max_cycle - df["time_in_cycles"], cap)
    return df


def select_signals(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 1e-3) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    op_cols     = [c for c in train.columns if c.startswith("op_setting_")]
    stds = train[sensor_cols].std()
    keep_sensors = [c for c in sensor_cols if stds[c] >= threshold]
    drop = [c for c in sensor_cols if c not in keep_sensors]
    return op_cols + keep_sensors, train.drop(columns=drop), test.drop(columns=drop)


def fit_scaler(df: pd.DataFrame, signal_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    arr = df[signal_cols].to_numpy()
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def apply_scaler(df: pd.DataFrame, signal_cols: list[str], mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df[signal_cols] = (df[signal_cols].to_numpy() - mean) / std
    return df


def make_windows_train(df: pd.DataFrame, signal_cols: list[str], window: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows over each engine. Pads early cycles by repeating the first row."""
    Xs, ys, units = [], [], []
    for unit, group in df.groupby("unit_number", sort=False):
        group = group.sort_values("time_in_cycles").reset_index(drop=True)
        values = group[signal_cols].to_numpy(dtype=np.float32)
        labels = group["rul"].to_numpy(dtype=np.float32)
        for i in range(len(group)):
            lo = i - window + 1
            if lo < 0:
                pad = np.repeat(values[:1], -lo, axis=0)
                win = np.concatenate([pad, values[: i + 1]], axis=0)
            else:
                win = values[lo : i + 1]
            Xs.append(win)
            ys.append(labels[i])
            units.append(unit)
    return np.stack(Xs), np.array(ys, dtype=np.float32), np.array(units)


def make_windows_test_last(df: pd.DataFrame, signal_cols: list[str], window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """One window per engine: the final `window` cycles."""
    Xs, units = [], []
    for unit, group in df.groupby("unit_number", sort=False):
        group = group.sort_values("time_in_cycles").reset_index(drop=True)
        values = group[signal_cols].to_numpy(dtype=np.float32)
        if len(group) < window:
            pad = np.repeat(values[:1], window - len(group), axis=0)
            win = np.concatenate([pad, values], axis=0)
        else:
            win = values[-window:]
        Xs.append(win)
        units.append(unit)
    return np.stack(Xs), np.array(units)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, n_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.posenc     = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(score.sum())


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":        float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "n":          int(len(y_true)),
    }


# ----------------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        optim.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float().to(DEVICE)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model_kind = os.getenv("MODEL", "lstm").lower()
    print(f"[advanced] Model: {model_kind} | Device: {DEVICE}")

    train_raw, test_raw, rul_truth = load_subset(SUBSET)
    train_raw = add_piecewise_rul(train_raw, cap=RUL_CAP)

    signal_cols, train_raw, test_raw = select_signals(train_raw, test_raw)
    print(f"[advanced] Using {len(signal_cols)} signals")

    mean, std = fit_scaler(train_raw, signal_cols)
    train_raw = apply_scaler(train_raw, signal_cols, mean, std)
    test_raw  = apply_scaler(test_raw,  signal_cols, mean, std)

    print("[advanced] Building train windows...")
    X_all, y_all, units_all = make_windows_train(train_raw, signal_cols, WINDOW)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    tr_idx, va_idx = next(gss.split(X_all, y_all, groups=units_all))
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]
    print(f"[advanced] Train windows: {X_tr.shape} | Val windows: {X_va.shape}")

    print("[advanced] Building test windows (one per engine)...")
    X_te, te_units = make_windows_test_last(test_raw, signal_cols, WINDOW)
    y_te = np.minimum(rul_truth["rul"].values, RUL_CAP).astype(np.float32)
    print(f"[advanced] Test windows: {X_te.shape}")

    n_features = X_tr.shape[2]
    if model_kind == "transformer":
        model = TransformerRegressor(n_features=n_features).to(DEVICE)
    else:
        model = LSTMRegressor(n_features=n_features).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val = math.inf
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dl, optim, loss_fn)
        val_pred = predict(model, X_va)
        val_metrics = evaluate(y_va, val_pred)
        history.append({"epoch": epoch, "train_mse": train_loss, **val_metrics})
        print(f"[advanced] epoch {epoch:03d} | train_mse={train_loss:.3f} | val_rmse={val_metrics['rmse']:.3f}")
        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[advanced] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred  = predict(model, X_va)
    val_metrics  = evaluate(y_va, val_pred)
    test_pred = predict(model, X_te)
    test_metrics = evaluate(y_te, test_pred)
    print(f"[advanced] Final validation: {val_metrics}")
    print(f"[advanced] NASA test set    : {test_metrics}")

    out_model   = DELIVERABLES / f"rul_{model_kind}.pt"
    out_metrics = DELIVERABLES / f"metrics_advanced_{model_kind}.json"
    out_preds   = DELIVERABLES / f"predictions_advanced_{model_kind}.csv"

    torch.save(model.state_dict(), out_model)
    pd.DataFrame({
        "unit_number": te_units,
        "rul_true":    y_te,
        "rul_pred":    test_pred,
    }).to_csv(out_preds, index=False)

    metrics = {
        "model": model_kind,
        "subset": SUBSET,
        "window": WINDOW,
        "rul_cap": RUL_CAP,
        "n_signals": n_features,
        "epochs_run": len(history),
        "validation": val_metrics,
        "test_nasa": test_metrics,
        "best_val_rmse": best_val,
        "history": history,
        "runtime_seconds": time.time() - t0,
    }
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[advanced] Saved model -> {out_model}")
    print(f"[advanced] Saved metrics -> {out_metrics}")
    print(f"[advanced] Saved predictions -> {out_preds}")
    print(f"[advanced] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
