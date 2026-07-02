"""Unit tests for DataProcessor in offline (synthetic) mode."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from data import DataProcessor

_CODE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture()
def config():
    with open(_CODE_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Force synthetic mode and a short window for fast tests
    cfg["data"]["use_synthetic_data"] = True
    cfg["data"]["start_date"] = "2022-01-01"
    cfg["data"]["end_date"] = "2022-06-30"
    cfg["data"]["train_start"] = "2022-01-01"
    cfg["data"]["train_end"] = "2022-04-30"
    cfg["data"]["test_start"] = "2022-05-01"
    cfg["data"]["test_end"] = "2022-06-30"
    return cfg


def test_synthetic_fetch(config):
    proc = DataProcessor(config)
    data = proc.fetch_data()
    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "tic"}
    assert expected_cols.issubset(set(data.columns))
    assert data["Close"].gt(0).all()
    # 25 investable assets + 1 macro factor (^VIX)
    assert data["tic"].nunique() == 26


def test_synthetic_reproducibility(config):
    d1 = DataProcessor(config).fetch_data()
    d2 = DataProcessor(config).fetch_data()
    np.testing.assert_array_almost_equal(d1["Close"].values, d2["Close"].values)


def test_indicators_and_split(config):
    proc = DataProcessor(config)
    train, test = proc.process_all()

    for col in ("macd", "rsi", "cci", "dx", "boll_ub", "turbulence"):
        assert col in train.columns, f"missing indicator column: {col}"

    assert not train.isnull().any().any(), "train data contains NaNs"
    assert not test.isnull().any().any(), "test data contains NaNs"

    # Macro factor must be excluded from the investable universe
    assert "^VIX" not in set(train["tic"].unique())

    # Chronological split with no overlap
    assert train["Date"].max() < test["Date"].min()


def test_turbulence_no_lookahead(config):
    """Warmup rows must be exactly zero (no future data used)."""
    proc = DataProcessor(config)
    proc.fetch_data()
    proc.calculate_technical_indicators()
    df = proc.add_turbulence_index()
    dates = np.sort(df["Date"].unique())
    early = df[df["Date"].isin(dates[:60])]
    assert (early["turbulence"] == 0.0).all()
