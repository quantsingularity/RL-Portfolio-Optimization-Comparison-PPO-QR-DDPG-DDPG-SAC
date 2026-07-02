"""
Portfolio Benchmark Strategies with Backtesting.

Strategies implemented:
    - Equal Weight (EW)
    - Mean-Variance Optimization (MVO)
    - Risk Parity (RP)
    - Minimum Volatility (MVP)
    - Momentum (MOM)
    - 60/40 Portfolio
    - All-Weather Portfolio (Ray Dalio)
    - Minimum Correlation (MC)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = _Path(__file__).resolve().parent.parent  # → code/
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))
del _sys, _Path, _CODE_DIR

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.045
DEFAULT_INITIAL_AMOUNT = 1_000_000
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_REBALANCE_FREQ = 20
DEFAULT_LOOKBACK_WINDOW = 60

VALID_STRATEGIES = {
    "equal_weight",
    "mvo",
    "risk_parity",
    "minimum_volatility",
    "momentum",
    "sixty_forty",
    "all_weather",
    "minimum_correlation",
}


# ---------------------------------------------------------------------------
# Strategy Engine
# ---------------------------------------------------------------------------


class BenchmarkStrategies:
    """
    Collection of portfolio allocation strategies.

    Parameters
    ----------
    returns_data : pd.DataFrame
        Daily returns, one column per asset.
    tickers : list[str]
        Asset tickers: must match ``returns_data`` columns.
    asset_classes : dict, optional
        Mapping of ticker → asset class string.
        Required by ``sixty_forty`` and ``all_weather``.
        Expected values: ``"equities"``, ``"fixed_income"``, ``"commodities"``.
    """

    def __init__(
        self,
        returns_data: pd.DataFrame,
        tickers: List[str],
        asset_classes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.returns_data = returns_data
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.asset_classes = asset_classes or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _uniform(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def _optimize(
        self,
        objective,
        initial: Optional[np.ndarray] = None,
        extra_constraints: Optional[list] = None,
    ) -> np.ndarray:
        """Generic SLSQP optimizer with a sum-to-one constraint and [0,1] bounds."""
        constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
        if extra_constraints:
            constraints.extend(extra_constraints)

        result = minimize(
            objective,
            initial if initial is not None else self._uniform(),
            method="SLSQP",
            bounds=[(0, 1)] * self.n_assets,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        return (
            result.x
            if result.success
            else (initial if initial is not None else self._uniform())
        )

    def _tickers_in_class(self, asset_class: str) -> List[int]:
        return [
            i
            for i, t in enumerate(self.tickers)
            if self.asset_classes.get(t) == asset_class
        ]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def equal_weight(self) -> np.ndarray:
        """Assign identical weight to every asset."""
        return self._uniform()

    def mean_variance_optimization(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> np.ndarray:
        """
        Maximize the Sharpe ratio (Markowitz MVO).

        Parameters
        ----------
        risk_free_rate : float
            Annualised risk-free rate used in the Sharpe calculation.
        """
        mean_ret = self.returns_data.mean().values
        cov = self.returns_data.cov().values
        daily_rf = risk_free_rate / TRADING_DAYS

        def neg_sharpe(w: np.ndarray) -> float:
            ret = w @ mean_ret
            vol = np.sqrt(w @ cov @ w)
            return -(ret - daily_rf) / vol if vol > 0 else 0.0

        return self._optimize(neg_sharpe)

    def risk_parity(self) -> np.ndarray:
        """
        Equalise each asset's contribution to total portfolio risk.
        """
        cov = self.returns_data.cov().values
        target = 1.0 / self.n_assets

        def objective(w: np.ndarray) -> float:
            vol = np.sqrt(w @ cov @ w)
            if vol == 0:
                return 0.0
            rc = (w * (cov @ w)) / vol  # risk contributions (fractional)
            return np.sum((rc / vol - target) ** 2)

        return self._optimize(objective)

    def minimum_volatility(self) -> np.ndarray:
        """Minimise portfolio variance."""
        cov = self.returns_data.cov().values

        def portfolio_variance(w: np.ndarray) -> float:
            return w @ cov @ w

        return self._optimize(portfolio_variance)

    def momentum(self, lookback: int = 20) -> np.ndarray:
        """
        Weight assets proportionally to their positive mean return
        over the last ``lookback`` periods.

        Parameters
        ----------
        lookback : int
            Number of recent periods used to measure momentum.
        """
        mom = self.returns_data.tail(lookback).mean().clip(lower=0).values
        total = mom.sum()
        return mom / total if total > 0 else self._uniform()

    def sixty_forty(self) -> np.ndarray:
        """
        Classic 60 / 40 portfolio: 60 % equities, 40 % fixed income.
        Weights within each class are equal.
        Requires ``asset_classes`` to contain ``"equities"`` / ``"fixed_income"`` entries.
        """
        weights = np.zeros(self.n_assets)
        eq_idx = self._tickers_in_class("equities")
        fi_idx = self._tickers_in_class("fixed_income")

        if eq_idx:
            weights[eq_idx] = 0.60 / len(eq_idx)
        if fi_idx:
            weights[fi_idx] = 0.40 / len(fi_idx)

        total = weights.sum()
        return weights / total if total > 0 else self._uniform()

    def all_weather(self) -> np.ndarray:
        """
        Ray Dalio All-Weather allocation:
            30 % equities | 40 % long bonds (TLT) | 15 % intermediate bonds (IEF)
            7.5 % gold (GC=F) | 7.5 % other commodities.

        Requires ``asset_classes`` with ``"equities"``, ``"fixed_income"``,
        and ``"commodities"`` entries.
        """
        weights = np.zeros(self.n_assets)

        eq_idx = self._tickers_in_class("equities")
        fi_idx = self._tickers_in_class("fixed_income")
        com_idx = self._tickers_in_class("commodities")

        # Equities: 30 %
        if eq_idx:
            weights[eq_idx] = 0.30 / len(eq_idx)

        # Fixed income: 55 % split between long / intermediate / other
        if fi_idx:
            long_idx = [i for i in fi_idx if "TLT" in self.tickers[i]]
            inter_idx = [i for i in fi_idx if "IEF" in self.tickers[i]]
            other_idx = [i for i in fi_idx if i not in long_idx and i not in inter_idx]

            if long_idx:
                weights[long_idx] = 0.40 / len(long_idx)
            if inter_idx:
                weights[inter_idx] = 0.15 / len(inter_idx)
            if other_idx:
                # Any remaining bonds share what's left of the 55 %
                used = (0.40 if long_idx else 0) + (0.15 if inter_idx else 0)
                weights[other_idx] = (0.55 - used) / len(other_idx)

        # Commodities: 15 % (gold gets half, rest split remainder)
        if com_idx:
            gold_idx = [i for i in com_idx if "GC=F" in self.tickers[i]]
            other_com = [i for i in com_idx if i not in gold_idx]

            if gold_idx:
                weights[gold_idx] = 0.075 / len(gold_idx)
            if other_com:
                weights[other_com] = 0.075 / len(other_com)
            elif not gold_idx:
                weights[com_idx] = 0.15 / len(com_idx)

        total = weights.sum()
        return weights / total if total > 0 else self._uniform()

    def minimum_correlation(self) -> np.ndarray:
        """
        Minimise the weighted-average pairwise correlation across assets.
        """
        corr = self.returns_data.corr().values

        def avg_correlation(w: np.ndarray) -> float:
            # w^T * C * w minus the diagonal (self-correlations = 1)
            return float(w @ corr @ w) - float(w @ w)

        return self._optimize(avg_correlation)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def get_weights(self, strategy: str, **kwargs) -> np.ndarray:
        """
        Return weights for a named strategy.

        Parameters
        ----------
        strategy : str
            One of :data:`VALID_STRATEGIES`.
        **kwargs
            Forwarded to the underlying strategy method.
        """
        dispatch = {
            "equal_weight": self.equal_weight,
            "mvo": self.mean_variance_optimization,
            "risk_parity": self.risk_parity,
            "minimum_volatility": self.minimum_volatility,
            "momentum": self.momentum,
            "sixty_forty": self.sixty_forty,
            "all_weather": self.all_weather,
            "minimum_correlation": self.minimum_correlation,
        }
        if strategy not in dispatch:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {sorted(dispatch)}"
            )
        return dispatch[strategy](**kwargs)


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------


class BacktestBenchmark:
    """
    Event-driven backtester for :class:`BenchmarkStrategies`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format market data with columns: ``Date``, ``tic``, ``Close``.
    initial_amount : float
        Starting portfolio value.
    transaction_cost_pct : float
        One-way cost applied to weight turnover on each rebalance.
    rebalance_freq : int
        Rebalancing cadence in trading days.
    asset_classes : dict, optional
        Passed through to :class:`BenchmarkStrategies`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = DEFAULT_INITIAL_AMOUNT,
        transaction_cost_pct: float = DEFAULT_TRANSACTION_COST,
        rebalance_freq: int = DEFAULT_REBALANCE_FREQ,
        asset_classes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.rebalance_freq = rebalance_freq
        self.asset_classes = asset_classes or {}

        self.dates = np.sort(df["Date"].unique())
        self.tickers = np.array(sorted(df["tic"].unique()))
        self.n_assets = len(self.tickers)

        # Pre-index prices for fast lookup: shape (n_dates, n_assets)
        self._price_matrix = self._build_price_matrix()

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _build_price_matrix(self) -> np.ndarray:
        """Pivot Close prices into a (dates × tickers) matrix."""
        pivot = (
            self.df.pivot(index="Date", columns="tic", values="Close")
            .reindex(index=self.dates, columns=self.tickers)
            .ffill()
            .bfill()
        )
        return pivot.values.astype(float)

    def _returns_slice(self, start: int, end: int) -> pd.DataFrame:
        """Return a DataFrame of daily returns for dates[start:end]."""
        prices = self._price_matrix[start:end]
        ret = np.diff(prices, axis=0) / prices[:-1]
        return pd.DataFrame(ret, columns=self.tickers)

    # ------------------------------------------------------------------
    # Core backtest loop
    # ------------------------------------------------------------------

    def backtest_strategy(
        self,
        strategy_name: str,
        lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        **strategy_kwargs,
    ) -> Dict:
        """
        Run a full backtest for one strategy.

        Parameters
        ----------
        strategy_name : str
            One of :data:`VALID_STRATEGIES`.
        lookback_window : int
            Number of historical trading days used to calibrate weights.
        risk_free_rate : float
            Annualised rate used when computing Sharpe / Sortino ratios.
        **strategy_kwargs
            Extra keyword arguments forwarded to the strategy method
            (e.g. ``lookback=20`` for momentum).

        Returns
        -------
        dict
            Performance metrics and portfolio value time-series.
        """
        if strategy_name not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy. Choose from: {sorted(VALID_STRATEGIES)}"
            )

        n_dates = len(self.dates)
        portfolio_value = float(self.initial_amount)
        weights = np.ones(self.n_assets) / self.n_assets

        portfolio_values = np.empty(n_dates)
        portfolio_values[0] = portfolio_value

        for i in range(1, n_dates):
            # ---- Rebalance ------------------------------------------------
            if i % self.rebalance_freq == 0 and i >= lookback_window:
                returns_df = self._returns_slice(i - lookback_window, i)

                engine = BenchmarkStrategies(
                    returns_data=returns_df,
                    tickers=list(self.tickers),
                    asset_classes=self.asset_classes,
                )
                new_weights = engine.get_weights(strategy_name, **strategy_kwargs)

                # Transaction costs on turnover
                turnover = np.abs(new_weights - weights).sum()
                portfolio_value -= (
                    turnover * portfolio_value * self.transaction_cost_pct
                )
                weights = new_weights

            # ---- Daily P&L ------------------------------------------------
            prev_prices = self._price_matrix[i - 1]
            curr_prices = self._price_matrix[i]

            valid = prev_prices > 0
            daily_ret = np.where(
                valid,
                (curr_prices - prev_prices) / prev_prices,
                0.0,
            )
            portfolio_value *= 1.0 + float(weights @ daily_ret)
            portfolio_values[i] = portfolio_value

        # ---- Metrics ------------------------------------------------------
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return self._compute_metrics(
            strategy_name=strategy_name,
            portfolio_values=portfolio_values,
            dates=self.dates,
            returns=returns,
            risk_free_rate=risk_free_rate,
        )

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def run_all(
        self,
        strategies: Optional[List[str]] = None,
        lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """
        Backtest multiple strategies and return a summary DataFrame.

        Parameters
        ----------
        strategies : list[str], optional
            Subset of :data:`VALID_STRATEGIES`. Defaults to all strategies.
        lookback_window : int
        risk_free_rate : float

        Returns
        -------
        pd.DataFrame
            One row per strategy with all performance metrics.
        """
        targets = strategies or sorted(VALID_STRATEGIES)
        results = []
        for name in targets:
            try:
                res = self.backtest_strategy(
                    name,
                    lookback_window=lookback_window,
                    risk_free_rate=risk_free_rate,
                )
                results.append(res)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Strategy '{name}' failed: {exc}")

        summary_cols = [
            "strategy",
            "final_value",
            "total_return_pct",
            "annual_return_pct",
            "volatility_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "cvar_5_pct",
        ]
        rows = [{k: v for k, v in r.items() if k in summary_cols} for r in results]
        return pd.DataFrame(rows).set_index("strategy")

    # ------------------------------------------------------------------
    # Metrics helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        strategy_name: str,
        portfolio_values: np.ndarray,
        dates: np.ndarray,
        returns: np.ndarray,
        risk_free_rate: float,
    ) -> Dict:
        n = len(returns)
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[
            0
        ]
        annual_return = (1 + total_return) ** (TRADING_DAYS / n) - 1 if n > 0 else 0.0
        volatility = returns.std() * np.sqrt(TRADING_DAYS)

        sharpe = (
            (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        )

        neg = returns[returns < 0]
        downside_vol = neg.std() * np.sqrt(TRADING_DAYS) if len(neg) > 0 else 0.0
        sortino = (
            (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
        )

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max()

        sorted_ret = np.sort(returns)
        cvar_cut = max(1, int(0.05 * len(sorted_ret)))
        cvar = sorted_ret[:cvar_cut].mean()

        return {
            "strategy": strategy_name,
            "portfolio_values": portfolio_values.tolist(),
            "dates": list(dates),
            "final_value": round(portfolio_values[-1], 2),
            "total_return_pct": round(total_return * 100, 4),
            "annual_return_pct": round(annual_return * 100, 4),
            "volatility_pct": round(volatility * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(-max_drawdown * 100, 4),
            "cvar_5_pct": round(cvar * 100, 4),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("benchmark_strategies.py loaded: all strategy and backtest classes ready.")
    print(f"Available strategies: {sorted(VALID_STRATEGIES)}")
