"""Shared helpers for the DRL test-suite."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_days: int = 120,
    tickers: tuple = ("AAA", "BBB", "CCC"),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a small long-format OHLCV + indicator panel suitable for
    ``PortfolioEnv``. Columns match what the environment consumes:
    Date, tic, Close, macd, rsi, cci, dx, boll_ub, turbulence.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for i, tic in enumerate(tickers):
        price = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n_days)))
        df = pd.DataFrame(
            {
                "Date": dates,
                "tic": tic,
                "Close": price.astype(np.float32),
                "macd": rng.normal(0, 1, n_days).astype(np.float32),
                "rsi": rng.uniform(20, 80, n_days).astype(np.float32),
                "cci": rng.normal(0, 100, n_days).astype(np.float32),
                "dx": rng.uniform(0, 50, n_days).astype(np.float32),
                "boll_ub": (price * 1.05).astype(np.float32),
                "turbulence": rng.uniform(0, 50, n_days).astype(np.float32),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "tic"])
