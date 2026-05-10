"""Project 11: NASA C-MAPSS Turbofan RUL - Baseline.

Random Forest regressor on hand-crafted lag/window features.

Per-cycle features (last `WINDOW` cycles of each engine):
    mean, std, min, max, slope (linear regression coef vs cycle)
applied to every (non-trivial) sensor + the three op_settings.

Pipeline
--------
1. Load FD001 train, test, RUL.
2. Drop low-variance sensors.
3. Build piece-wise linear RUL labels with cap = 125.
4. Build window features at every cycle (training: every cycle of every engine).
5. Fit RandomForestRegressor.
6. Score on held-out engines (within train) and on the NASA test set.
7. Save model + metrics to deliverables/.

Run
---
    python src/model_baseline.py

Author: Sandeep Grover, Independent Research.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


HERE          = Path(__file__).resolve().parent
PROJECT_DIR   = HERE.parent
DATA_DIR      = PROJECT_DIR / "data"
CMAPS_DIR     = DATA_DIR / "CMaps" if (DATA_DIR / "CMaps").exists() else DATA_DIR
DELIVERABLES  = PROJECT_DIR / "deliverables"
DELIVERABLES.mkdir(parents=True, exist_ok=True)

SUBSET     = "FD001"
WINDOW     = 30
RUL_CAP    = 125
RANDOM_SEED = 42

COLS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ----------------------------------------------------------------------------
# Data loading
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


def drop_low_variance(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 1e-3) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    stds = train[sensor_cols].std()
    keep = [c for c in sensor_cols if stds[c] >= threshold]
    drop = [c for c in sensor_cols if c not in keep]
    return keep, train.drop(columns=drop), test.drop(columns=drop)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------

def _slope(arr: np.ndarray) -> float:
    """Linear regression slope of arr vs an evenly-spaced index. Returns 0 for short arrays."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (arr - y_mean)).sum() / denom)


def _window_stats(arr: np.ndarray) -> tuple[float, float, float, float, float]:
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), _slope(arr)


def build_window_features(df: pd.DataFrame, signal_cols: list[str], window: int = WINDOW, label_col: str | None = "rul") -> pd.DataFrame:
    """For every cycle of every engine, extract stats over the trailing `window` cycles."""
    rows = []
    for unit, group in df.groupby("unit_number", sort=False):
        group = group.sort_values("time_in_cycles").reset_index(drop=True)
        values = group[signal_cols].to_numpy()
        labels = group[label_col].to_numpy() if label_col else None
        cycles = group["time_in_cycles"].to_numpy()
        for i in range(len(group)):
            lo = max(0, i - window + 1)
            win = values[lo : i + 1]
            feat: dict = {"unit_number": unit, "time_in_cycles": int(cycles[i])}
            for j, col in enumerate(signal_cols):
                m, s, mn, mx, sl = _window_stats(win[:, j])
                feat[f"{col}_mean"]  = m
                feat[f"{col}_std"]   = s
                feat[f"{col}_min"]   = mn
                feat[f"{col}_max"]   = mx
                feat[f"{col}_slope"] = sl
            if labels is not None:
                feat["rul"] = float(labels[i])
            rows.append(feat)
    return pd.DataFrame(rows)


def last_cycle_per_unit(feature_df: pd.DataFrame) -> pd.DataFrame:
    return feature_df.sort_values(["unit_number", "time_in_cycles"]).groupby("unit_number", as_index=False).tail(1)


# ----------------------------------------------------------------------------
# Metrics - including the NASA asymmetric scoring function
# ----------------------------------------------------------------------------

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Saxena 2008. Penalises late predictions (over-estimating RUL) more heavily."""
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
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print(f"[baseline] Loading {SUBSET} from {CMAPS_DIR}")
    train_raw, test_raw, rul_truth = load_subset(SUBSET)

    print(f"[baseline] Train rows={len(train_raw)}, Test rows={len(test_raw)}, RUL rows={len(rul_truth)}")

    train_raw = add_piecewise_rul(train_raw, cap=RUL_CAP)

    keep_sensors, train_raw, test_raw = drop_low_variance(train_raw, test_raw)
    op_cols = [c for c in train_raw.columns if c.startswith("op_setting_")]
    signal_cols = op_cols + keep_sensors
    print(f"[baseline] Kept {len(keep_sensors)} sensors + {len(op_cols)} op_settings = {len(signal_cols)} signals")

    print("[baseline] Building window features (train)...")
    train_feat = build_window_features(train_raw, signal_cols=signal_cols, window=WINDOW, label_col="rul")
    print(f"[baseline] Train feature matrix: {train_feat.shape}")

    print("[baseline] Building window features (test, last cycle of each engine)...")
    test_feat_all  = build_window_features(test_raw, signal_cols=signal_cols, window=WINDOW, label_col=None)
    test_feat_last = last_cycle_per_unit(test_feat_all).reset_index(drop=True)
    test_feat_last = test_feat_last.sort_values("unit_number").reset_index(drop=True)
    test_feat_last["rul"] = rul_truth["rul"].clip(upper=RUL_CAP).values
    print(f"[baseline] Test feature matrix (one row per engine): {test_feat_last.shape}")

    feat_cols = [c for c in train_feat.columns if c not in ("unit_number", "time_in_cycles", "rul")]
    X_train_full = train_feat[feat_cols].values
    y_train_full = train_feat["rul"].values
    groups_full  = train_feat["unit_number"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss.split(X_train_full, y_train_full, groups=groups_full))

    print("[baseline] Fitting RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_full[train_idx], y_train_full[train_idx])

    val_pred = model.predict(X_train_full[val_idx])
    val_metrics = evaluate(y_train_full[val_idx], val_pred)
    print(f"[baseline] Validation: {val_metrics}")

    test_pred = model.predict(test_feat_last[feat_cols].values)
    test_metrics = evaluate(test_feat_last["rul"].values, test_pred)
    print(f"[baseline] NASA test set: {test_metrics}")

    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top20 = importances.head(20).to_dict()

    out_model   = DELIVERABLES / "rul_random_forest.pkl"
    out_metrics = DELIVERABLES / "metrics_baseline.json"
    out_preds   = DELIVERABLES / "predictions_baseline.csv"

    joblib.dump(model, out_model)
    pd.DataFrame({
        "unit_number": test_feat_last["unit_number"].values,
        "rul_true":    test_feat_last["rul"].values,
        "rul_pred":    test_pred,
    }).to_csv(out_preds, index=False)

    metrics = {
        "subset": SUBSET,
        "window": WINDOW,
        "rul_cap": RUL_CAP,
        "n_signals": len(signal_cols),
        "validation": val_metrics,
        "test_nasa": test_metrics,
        "top20_features": top20,
        "runtime_seconds": time.time() - t0,
    }
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[baseline] Saved model -> {out_model}")
    print(f"[baseline] Saved metrics -> {out_metrics}")
    print(f"[baseline] Saved predictions -> {out_preds}")
    print(f"[baseline] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
