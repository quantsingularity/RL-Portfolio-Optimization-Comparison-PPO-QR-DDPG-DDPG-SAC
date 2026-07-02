"""
Data preprocessing and feature engineering module.

This module handles:
1. Data fetching from Yahoo Finance
2. Technical indicator calculation
3. Feature engineering for the RL environment
4. Data cleaning and normalization
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance is optional for offline mode
    yf = None

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent  # → code/


class DataProcessor:
    """Download, clean, and feature-engineer financial data for RL training."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.valid_tickers: List[str] = []  # tickers with sufficient data

    # ------------------------------------------------------------------ #
    # Step 1 – Data fetching                                               #
    # ------------------------------------------------------------------ #

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data for all configured assets in a single batch call.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: Date, Open, High, Low,
            Close, Volume, tic.
        """
        print("Fetching data from Yahoo Finance (batch download)...")

        all_assets: List[str] = []
        for asset_class in [
            "equities",
            "cryptocurrencies",
            "commodities",
            "fixed_income",
        ]:
            all_assets.extend(self.config["data"]["assets"].get(asset_class, []))

        # Macro factors are downloaded separately and NOT included in the
        # investable universe (e.g. ^VIX cannot be traded).
        macro_factors: List[str] = self.config["data"].get("macro_factors", [])

        start_date = self.config["data"]["start_date"]
        end_date = self.config["data"]["end_date"]

        # ---- Offline / synthetic mode ------------------------------------- #
        # If explicitly requested, or if the live download fails (e.g. no
        # network access), fall back to reproducible synthetic OHLCV data so
        # the full pipeline remains runnable end-to-end without Yahoo Finance.
        if self.config["data"].get("use_synthetic_data", False):
            print("  use_synthetic_data=True: generating synthetic OHLCV data.")
            return self._generate_synthetic_data(all_assets, macro_factors)

        try:
            if yf is None:
                raise RuntimeError("yfinance is not installed")
            raw = yf.download(
                all_assets,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )
            if raw is None or len(raw) == 0:
                raise RuntimeError("empty response from Yahoo Finance")
        except Exception as exc:  # pragma: no cover - network dependent
            print(
                f"  [WARN] Live data download failed ({exc}). "
                "Falling back to synthetic data."
            )
            return self._generate_synthetic_data(all_assets, macro_factors)

        data_list: List[pd.DataFrame] = []
        failed: List[str] = []

        for ticker in all_assets:
            try:
                if len(all_assets) == 1:
                    df_t = raw.copy()
                else:
                    df_t = raw[ticker].copy()

                df_t = df_t.dropna(subset=["Close"])

                if len(df_t) < 30:
                    print(f"  [SKIP] {ticker}: insufficient data ({len(df_t)} rows)")
                    failed.append(ticker)
                    continue

                df_t = df_t.reset_index()
                # yfinance ≥0.2.x may name the date index 'Datetime'
                if "Datetime" in df_t.columns and "Date" not in df_t.columns:
                    df_t = df_t.rename(columns={"Datetime": "Date"})
                # Strip timezone so date comparisons with naive config strings work
                if pd.api.types.is_datetime64tz_dtype(df_t["Date"]):
                    df_t["Date"] = df_t["Date"].dt.tz_localize(None)
                df_t["tic"] = ticker
                data_list.append(
                    df_t[["Date", "Open", "High", "Low", "Close", "Volume", "tic"]]
                )

            except Exception as exc:
                print(f"  [SKIP] {ticker}: {exc}")
                failed.append(ticker)

        if failed:
            print(f"  Dropped {len(failed)} ticker(s): {failed}")

        self.valid_tickers = [t for t in all_assets if t not in failed]

        # ---- Macro factors (separate download, not in portfolio) ----------- #
        if macro_factors:
            macro_raw = yf.download(
                macro_factors,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )
            for ticker in macro_factors:
                try:
                    df_t = (
                        macro_raw[ticker] if len(macro_factors) > 1 else macro_raw
                    ).copy()
                    df_t = df_t.dropna(subset=["Close"]).reset_index()
                    if "Datetime" in df_t.columns and "Date" not in df_t.columns:
                        df_t = df_t.rename(columns={"Datetime": "Date"})
                    if pd.api.types.is_datetime64tz_dtype(df_t["Date"]):
                        df_t["Date"] = df_t["Date"].dt.tz_localize(None)
                    df_t["tic"] = ticker
                    data_list.append(
                        df_t[["Date", "Open", "High", "Low", "Close", "Volume", "tic"]]
                    )
                    self.valid_tickers.append(ticker)
                except Exception as exc:
                    print(f"  [SKIP macro] {ticker}: {exc}")

        self.data = (
            pd.concat(data_list, ignore_index=True)
            .sort_values(["Date", "tic"])
            .reset_index(drop=True)
        )
        self.data["Date"] = pd.to_datetime(self.data["Date"])

        print(
            f"Data fetched: {len(self.data):,} rows, "
            f"{self.data['tic'].nunique()} tickers"
        )
        return self.data

    # ------------------------------------------------------------------ #
    # Offline synthetic data generator                                     #
    # ------------------------------------------------------------------ #

    def _generate_synthetic_data(
        self,
        investable: List[str],
        macro_factors: List[str],
    ) -> pd.DataFrame:
        """
        Generate reproducible synthetic OHLCV data for every configured
        ticker using a geometric-Brownian-motion price process.

        This keeps the whole training / evaluation / serving pipeline runnable
        end-to-end in environments without access to Yahoo Finance. The data
        is clearly synthetic and seeded for reproducibility.
        """
        start_date = pd.to_datetime(self.config["data"]["start_date"])
        end_date = pd.to_datetime(self.config["data"]["end_date"])
        # Business-day calendar keeps the series realistic (no weekends).
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)

        rng = np.random.default_rng(self.config["data"].get("synthetic_seed", 42))
        all_tickers = list(investable) + list(macro_factors)
        frames: List[pd.DataFrame] = []

        for idx, ticker in enumerate(all_tickers):
            # Per-ticker drift / volatility so assets are not identical.
            mu = rng.uniform(0.00002, 0.0006)
            sigma = rng.uniform(0.008, 0.03)
            start_price = float(rng.uniform(20.0, 500.0))

            shocks = rng.normal(mu, sigma, size=n)
            log_path = np.cumsum(shocks)
            close = start_price * np.exp(log_path)

            # Build OHLC around the close with a small intraday range.
            intraday = np.abs(rng.normal(0.0, sigma, size=n)) * close
            open_ = close * (1.0 + rng.normal(0.0, sigma / 2, size=n))
            high = np.maximum(open_, close) + intraday
            low = np.minimum(open_, close) - intraday
            low = np.clip(low, 1e-3, None)
            volume = rng.integers(1_000_000, 20_000_000, size=n).astype(float)

            frames.append(
                pd.DataFrame(
                    {
                        "Date": dates,
                        "Open": open_.astype(np.float32),
                        "High": high.astype(np.float32),
                        "Low": low.astype(np.float32),
                        "Close": close.astype(np.float32),
                        "Volume": volume,
                        "tic": ticker,
                    }
                )
            )

        self.valid_tickers = list(all_tickers)
        self.data = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["Date", "tic"])
            .reset_index(drop=True)
        )
        self.data["Date"] = pd.to_datetime(self.data["Date"])

        print(
            f"Synthetic data generated: {len(self.data):,} rows, "
            f"{self.data['tic'].nunique()} tickers over {n} trading days"
        )
        return self.data

    # ------------------------------------------------------------------ #
    # Step 2 – Technical indicators                                         #
    # ------------------------------------------------------------------ #

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Compute MACD, RSI, CCI, DX, and Bollinger Bands for every ticker."""
        print("Calculating technical indicators...")

        processed_list: List[pd.DataFrame] = []

        for ticker in self.data["tic"].unique():
            t = self.data[self.data["tic"] == ticker].copy().sort_values("Date")

            # MACD
            exp1 = t["Close"].ewm(span=12, adjust=False).mean()
            exp2 = t["Close"].ewm(span=26, adjust=False).mean()
            t["macd"] = exp1 - exp2
            t["macd_signal"] = t["macd"].ewm(span=9, adjust=False).mean()
            t["macd_diff"] = t["macd"] - t["macd_signal"]

            # RSI
            delta = t["Close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            t["rsi"] = 100 - (100 / (1 + rs))

            # CCI
            tp = (t["High"] + t["Low"] + t["Close"]) / 3
            sma = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            t["cci"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))

            # DX
            h_diff = t["High"].diff()
            l_diff = -t["Low"].diff()
            pos_dm = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
            neg_dm = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)

            tr = pd.concat(
                [
                    t["High"] - t["Low"],
                    (t["High"] - t["Close"].shift()).abs(),
                    (t["Low"] - t["Close"].shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(14).mean().replace(0, np.nan)
            pos_di = 100 * pos_dm.rolling(14).mean() / atr
            neg_di = 100 * neg_dm.rolling(14).mean() / atr
            di_sum = (pos_di + neg_di).replace(0, np.nan)
            t["dx"] = 100 * (pos_di - neg_di).abs() / di_sum

            # Bollinger Bands
            sma20 = t["Close"].rolling(20).mean()
            std20 = t["Close"].rolling(20).std()
            t["boll_ub"] = sma20 + 2 * std20
            t["boll_lb"] = sma20 - 2 * std20

            processed_list.append(t)

        # Fill NaNs per-ticker *before* concatenating so we never propagate
        # one asset's values into another asset's missing rows.
        filled_list = [t.sort_values("Date").ffill().bfill() for t in processed_list]
        self.processed_data = (
            pd.concat(filled_list, ignore_index=True)
            .sort_values(["Date", "tic"])
            .reset_index(drop=True)
        )

        print("Technical indicators calculated successfully")
        return self.processed_data

    # ------------------------------------------------------------------ #
    # Step 3 – Turbulence index (look-ahead-free)                          #
    # ------------------------------------------------------------------ #

    def add_turbulence_index(self) -> pd.DataFrame:
        """
        Compute the Mahalanobis-distance turbulence index.

        Fix: uses an *expanding* covariance matrix computed only on data
        up to (but not including) the current date, removing the look-ahead
        bias present in the original implementation.

        A minimum warmup of 60 trading days is required before the
        Mahalanobis distance can be estimated; earlier rows receive 0.
        """
        print("Calculating turbulence index (expanding window, no look-ahead)...")

        df = self.processed_data.copy()
        df["returns"] = df.groupby("tic")["Close"].pct_change().fillna(0.0)

        returns_pivot = df.pivot_table(
            index="Date", columns="tic", values="returns", aggfunc="first"
        ).fillna(0.0)

        dates = returns_pivot.index.tolist()
        warmup = 60
        turb_list = []

        for i, date in enumerate(dates):
            if i < warmup:
                turb_list.append({"Date": date, "turbulence": 0.0})
                continue

            # Only look at history BEFORE current date (expanding, no leakage)
            hist = returns_pivot.iloc[:i]
            hist_mean = hist.mean()
            hist_cov = hist.cov()

            curr = returns_pivot.iloc[i].values

            try:
                diff = curr - hist_mean.values
                inv_cov = np.linalg.pinv(hist_cov.values)
                turb = float(diff @ inv_cov @ diff)
            except Exception:
                turb = 0.0

            turb_list.append({"Date": date, "turbulence": turb})

        turb_df = pd.DataFrame(turb_list)
        self.processed_data = df.merge(turb_df, on="Date", how="left")
        # Any rows whose date failed to align (dtype/timezone edge cases)
        # would otherwise carry NaN turbulence into the environment.
        self.processed_data["turbulence"] = self.processed_data["turbulence"].fillna(
            0.0
        )

        print("Turbulence index calculated (expanding window)")
        return self.processed_data

    # ------------------------------------------------------------------ #
    # Step 4 – Train / test split                                          #
    # ------------------------------------------------------------------ #

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_start = pd.to_datetime(self.config["data"]["train_start"])
        train_end = pd.to_datetime(self.config["data"]["train_end"])
        test_start = pd.to_datetime(self.config["data"]["test_start"])
        test_end = pd.to_datetime(self.config["data"]["test_end"])

        df = self.processed_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Keep only investable tickers (exclude macro-only factors from the
        # portfolio universe if they cannot be held as positions)
        investable = [
            t
            for t in self.valid_tickers
            if t not in self.config["data"].get("macro_factors", [])
        ]
        df = df[df["tic"].isin(investable)]

        train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
        test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

        print(
            f"Train: {train['Date'].min().date()} → {train['Date'].max().date()} "
            f"({len(train):,} rows)"
        )
        print(
            f"Test : {test['Date'].min().date()} → {test['Date'].max().date()} "
            f"({len(test):,} rows)"
        )
        return train, test

    # ------------------------------------------------------------------ #
    # Orchestrator                                                          #
    # ------------------------------------------------------------------ #

    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run all processing steps and return (train_data, test_data)."""
        self.fetch_data()
        self.calculate_technical_indicators()
        self.add_turbulence_index()
        return self.split_data()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    config_path = _ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processor = DataProcessor(config)
    train_data, test_data = processor.process_all()

    print("\nData processing complete!")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape:  {test_data.shape}")
