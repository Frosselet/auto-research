"""Generate a small deterministic synthetic OHLCV CSV for the reference recipe."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n = 1200
    dates = pd.bdate_range("2018-01-01", periods=n)
    # AR(1) log-return with mild vol clustering and a weak momentum effect
    eps = rng.standard_normal(n) * 0.01
    lr = np.zeros(n)
    for i in range(1, n):
        lr[i] = 0.05 * lr[i - 1] + eps[i]
    close = 100 * np.exp(np.cumsum(lr))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * 0.005)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * 0.005)
    volume = rng.integers(1_000_000, 5_000_000, size=n)

    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    out = Path(__file__).parent / "data.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out} ({n} rows)")


if __name__ == "__main__":
    main()
