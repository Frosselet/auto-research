"""Evaluate a direction classifier by computing out-of-sample daily Sharpe.

Contract (enforced by auto_research):
  - Reads AUTORESEARCH_DATA_PATH and AUTORESEARCH_ARTIFACT_DIR from env.
  - Writes {"metric": <sharpe>} into AUTORESEARCH_ARTIFACT_DIR/metric.json.

This file is NOT edited by the loop; it's the stable scoring target.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()
    out["ret_1"] = ret
    out["ret_5"] = df["close"].pct_change(5)
    out["ret_20"] = df["close"].pct_change(20)
    out["vol_20"] = ret.rolling(20).std()
    out["mom_10"] = (df["close"] / df["close"].shift(10)) - 1.0
    out["rng"] = (df["high"] - df["low"]) / df["close"]
    return out


def main() -> None:
    data_path = Path(os.environ["AUTORESEARCH_DATA_PATH"])
    artifact_dir = Path(os.environ["AUTORESEARCH_ARTIFACT_DIR"])

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    with (artifact_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    meta = json.loads((artifact_dir / "features.json").read_text())
    feature_cols = meta["features"]
    cut = int(meta["cut"])

    feats = build_features(df)
    ret = df["close"].pct_change().shift(-1)  # next-day return aligned to signal date

    X_oos = feats.iloc[cut:].copy()
    # Build any missing columns the loop-edited trainer may have added; align ordering.
    for col in feature_cols:
        if col not in X_oos.columns:
            X_oos[col] = 0.0
    X_oos = X_oos[feature_cols]
    mask = ~X_oos.isna().any(axis=1) & ~ret.iloc[cut:].isna()
    X_used = X_oos[mask].values
    r_oos = ret.iloc[cut:][mask].values

    proba = model.predict_proba(X_used)[:, 1]
    signal = np.where(proba > 0.5, 1.0, -1.0)
    pnl = signal * r_oos

    sharpe = float(np.sqrt(252) * pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0

    (artifact_dir / "metric.json").write_text(json.dumps({"metric": sharpe}))


if __name__ == "__main__":
    main()
