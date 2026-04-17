"""Train a gradient-boosted classifier that predicts next-day direction.

Contract (enforced by auto_research):
  - Reads AUTORESEARCH_DATA_PATH (a CSV with date/open/high/low/close/volume).
  - Writes model.pkl + features.json into AUTORESEARCH_ARTIFACT_DIR.

The autoresearch loop will edit this file. Keep it self-contained.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


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
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    feats = build_features(df)
    y = (df["close"].shift(-1) > df["close"]).astype(int)

    # In-sample window = first 70% of rows. Eval uses remaining 30%.
    n = len(df)
    cut = int(n * 0.7)
    X_train = feats.iloc[:cut].dropna()
    y_train = y.iloc[X_train.index]

    feature_cols = list(X_train.columns)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train.values, y_train.values)

    with (artifact_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    (artifact_dir / "features.json").write_text(json.dumps({"features": feature_cols, "cut": cut}))


if __name__ == "__main__":
    main()
