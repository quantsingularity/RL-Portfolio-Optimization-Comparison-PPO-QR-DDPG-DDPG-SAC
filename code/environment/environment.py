"""
Custom Portfolio Environment for RL Training.

This module implements the Markov Decision Process (MDP) formulation
with risk-aware reward function.

"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


class PortfolioEnv(gym.Env):
    """
    Custom Gymnasium Environment for Portfolio Optimization.

    State Space
    -----------
    [normalised_portfolio_value, *portfolio_weights, *per_asset_features]

    Per-asset features (6 per asset): Close/100, MACD/10, RSI/100,
    CCI/100, DX/100, BollingerUB/100.

    Action Space
    ------------
    Continuous vector in [-1, 1]^n.  Actions are passed through softmax
    so they represent *target portfolio weights* directly.  This is more
    learnable than the original delta-weight scheme.

    Reward
    ------
    log_return  -  lambda * current_drawdown  -  transaction_cost_fraction
    Computed on every step including the terminal step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 1_000_000,
        transaction_cost_pct: float = 0.001,
        max_drawdown_penalty: float = 0.5,
        hmax: float = 0.30,  # maximum weight per asset (fraction, not shares)
        print_verbosity: int = 5,
        turbulence_threshold: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Long-format DataFrame with columns: Date, tic, Close,
            macd, rsi, cci, dx, boll_ub.
        initial_amount : float
            Starting portfolio value in USD.
        transaction_cost_pct : float
            One-way transaction cost applied to weight turnover.
        max_drawdown_penalty : float
            Lambda coefficient multiplying the drawdown term in the reward.
        hmax : float
            Maximum fraction of portfolio that may be allocated to any single
            asset (e.g. 0.30 → 30 % cap).
        print_verbosity : int
            Print a progress line every this many steps during render().
        turbulence_threshold : float or None
            If set, positions are cut to cash when the turbulence index
            exceeds this value.
        """
        super().__init__()

        self.df = df.copy().sort_values(["Date", "tic"]).reset_index(drop=True)
        self.initial_amount = float(initial_amount)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.max_drawdown_penalty = float(max_drawdown_penalty)
        self.hmax = float(hmax)
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold

        # Sorted unique dates and tickers for reproducible indexing
        self.dates = np.sort(self.df["Date"].unique())
        self.tickers = np.array(sorted(self.df["tic"].unique()))
        self.n_stocks = len(self.tickers)
        self.n_dates = len(self.dates)

        # ------------------------------------------------------------------ #
        # Pre-build price matrix  shape: (n_dates, n_stocks)                  #
        # This replaces all per-step DataFrame filtering.                      #
        # ------------------------------------------------------------------ #
        self._price_matrix = self._build_price_matrix()
        self._feature_matrix = self._build_feature_matrix()

        # State: [portfolio_value_norm, *weights, *features_per_asset]
        self._n_features_per_asset = 6  # Close, MACD, RSI, CCI, DX, BollUB
        self.state_dim = (
            1  # normalised portfolio value
            + self.n_stocks  # current weights
            + self.n_stocks * self._n_features_per_asset  # market features
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Episode state (initialised properly in reset)
        self.current_step: int = 0
        self.portfolio_value: float = self.initial_amount
        self.portfolio_weights: np.ndarray = np.zeros(self.n_stocks, dtype=np.float32)
        self.max_portfolio_value: float = self.initial_amount

        self.portfolio_values: list = [self.initial_amount]
        self.portfolio_returns: list = [0.0]
        self.actions_memory: list = []
        self.date_memory: list = [self.dates[0]]

        print(f"PortfolioEnv: {self.n_stocks} assets, state_dim={self.state_dim}")

    # ------------------------------------------------------------------ #
    # Pre-processing helpers (called once at __init__)                     #
    # ------------------------------------------------------------------ #

    def _build_price_matrix(self) -> np.ndarray:
        """Return array of shape (n_dates, n_stocks) with Close prices."""
        pivot = (
            self.df.pivot(index="Date", columns="tic", values="Close")
            .reindex(index=self.dates, columns=self.tickers)
            .ffill()
            .bfill()
        )
        return pivot.values.astype(np.float32)

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Return array of shape (n_dates, n_stocks, n_features_per_asset).
        Features (in order): Close/100, MACD/10, RSI/100, CCI/100, DX/100, BollUB/100.
        """
        feature_cols = ["Close", "macd", "rsi", "cci", "dx", "boll_ub"]
        divisors = np.array([100, 10, 100, 100, 100, 100], dtype=np.float32)

        matrices = []
        for col, div in zip(feature_cols, divisors):
            if col in self.df.columns:
                pivot = (
                    self.df.pivot(index="Date", columns="tic", values=col)
                    .reindex(index=self.dates, columns=self.tickers)
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                )
                matrices.append((pivot.values / div).astype(np.float32))
            else:
                matrices.append(
                    np.zeros((self.n_dates, self.n_stocks), dtype=np.float32)
                )

        # stack → (n_dates, n_features, n_stocks) → transpose → (n_dates, n_stocks, n_features)
        return np.stack(matrices, axis=1).transpose(0, 2, 1)

    # ------------------------------------------------------------------ #
    # Gymnasium API                                                         #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to the start of the episode."""
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.max_portfolio_value = self.initial_amount
        self.portfolio_weights = np.zeros(self.n_stocks, dtype=np.float32)

        self.portfolio_values = [self.initial_amount]
        self.portfolio_returns = [0.0]
        self.actions_memory = []
        self.date_memory = [self.dates[0]]

        return self._get_state(), {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        Parameters
        ----------
        actions : np.ndarray  shape (n_stocks,)
            Raw policy output in [-1, 1].  Converted to target weights
            via softmax so weights are non-negative and sum to 1.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        terminated = self.current_step >= self.n_dates - 1
        truncated = False

        # ------ Convert actions → target weights (softmax projection) ------- #
        # Softmax on raw logits gives a proper simplex point.
        exp_a = np.exp(actions - actions.max())  # numerically stable
        new_weights = exp_a / exp_a.sum()

        # Apply per-asset cap (hmax) with an iterative capped-simplex
        # projection. A single clip-then-renormalise pass can push weights
        # back above the cap; iterating redistributes the excess only among
        # uncapped assets until all constraints hold.
        if self.hmax < 1.0:
            new_weights = self._project_capped_simplex(new_weights, self.hmax)

        # Turbulence guard: liquidate to equal-weight if threshold exceeded
        if self.turbulence_threshold is not None and self.current_step < self.n_dates:
            turb_col = "turbulence"
            if turb_col in self.df.columns:
                current_date = self.dates[self.current_step]
                turb_val = self.df.loc[self.df["Date"] == current_date, turb_col].mean()
                if turb_val > self.turbulence_threshold:
                    new_weights = np.full(
                        self.n_stocks, 1.0 / self.n_stocks, dtype=np.float32
                    )

        # ------ Transaction costs ------------------------------------------- #
        weight_delta = np.abs(new_weights - self.portfolio_weights).sum()
        transaction_cost = (
            weight_delta * self.portfolio_value * self.transaction_cost_pct
        )

        self.portfolio_weights = new_weights.astype(np.float32)

        # ------ Portfolio return for this step -------------------------------- #
        # Use pre-built price matrix: O(n_stocks) vector op, not DataFrame scan
        prev_prices = self._price_matrix[self.current_step]
        next_step = min(self.current_step + 1, self.n_dates - 1)
        next_prices = self._price_matrix[next_step]

        valid = prev_prices > 0
        asset_returns = np.where(valid, (next_prices - prev_prices) / prev_prices, 0.0)
        portfolio_return = float(self.portfolio_weights @ asset_returns)

        # ------ Update portfolio value --------------------------------------- #
        old_value = self.portfolio_value
        self.portfolio_value = old_value * (1.0 + portfolio_return) - transaction_cost
        self.portfolio_value = max(self.portfolio_value, 1.0)  # floor at $1

        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

        # ------ Reward -------------------------------------------------------- #
        # Computed on every step including terminal: fixes silent zero reward
        log_return = np.log(self.portfolio_value / old_value) if old_value > 0 else 0.0
        current_drawdown = (
            (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if self.max_portfolio_value > 0
            else 0.0
        )
        reward = float(
            log_return
            - self.max_drawdown_penalty * current_drawdown
            - (transaction_cost / self.initial_amount)
        )

        # ------ Bookkeeping -------------------------------------------------- #
        self.portfolio_values.append(self.portfolio_value)
        self.portfolio_returns.append(portfolio_return)
        self.actions_memory.append(actions)
        self.current_step += 1
        # Always append a date so date_memory and portfolio_values stay in sync.
        # At the terminal step current_step == n_dates; clamp to last valid index.
        self.date_memory.append(self.dates[min(self.current_step, self.n_dates - 1)])

        next_state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "transaction_cost": transaction_cost,
            "date": self.dates[min(self.current_step, self.n_dates - 1)],
        }
        return next_state, reward, terminated, truncated, info

    def render(self) -> None:
        if self.print_verbosity > 0 and self.current_step % self.print_verbosity == 0:
            print(
                f"Step {self.current_step:>5} | "
                f"Portfolio Value: ${self.portfolio_value:>14,.2f}"
            )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _project_capped_simplex(self, weights: np.ndarray, cap: float) -> np.ndarray:
        """
        Project weights onto the capped simplex:
            w_i in [0, cap],  sum(w) = 1.

        Iteratively caps violating assets and redistributes the remaining
        mass proportionally among uncapped assets. If the cap is infeasible
        (cap * n_stocks < 1), returns the closest feasible allocation
        (all assets at cap, renormalised).
        """
        n = len(weights)
        if cap * n < 1.0:
            # Infeasible cap: best effort: uniform at renormalised cap.
            return np.full(n, 1.0 / n, dtype=np.float32)

        w = np.clip(weights.astype(np.float64), 0.0, None)
        total = w.sum()
        w = w / total if total > 0 else np.full(n, 1.0 / n)

        capped = np.zeros(n, dtype=bool)
        for _ in range(n):
            over = (w > cap) & (~capped)
            if not over.any():
                break
            w[over] = cap
            capped |= over
            free = ~capped
            residual = 1.0 - w[capped].sum()
            free_sum = w[free].sum()
            if free_sum <= 0 or not free.any():
                # Distribute residual equally among free assets (or stop).
                if free.any():
                    w[free] = residual / free.sum()
                break
            w[free] = w[free] * (residual / free_sum)

        return w.astype(np.float32)

    def _get_state(self) -> np.ndarray:
        step = min(self.current_step, self.n_dates - 1)

        normalised_value = self.portfolio_value / self.initial_amount
        weights = self.portfolio_weights.copy()

        # Feature matrix already pre-computed: shape (n_stocks, n_features)
        features_2d = self._feature_matrix[step]  # (n_stocks, 6)
        features_flat = features_2d.ravel()  # (n_stocks * 6,)

        state = np.concatenate([[normalised_value], weights, features_flat]).astype(
            np.float32
        )
        return state

    # ------------------------------------------------------------------ #
    # Metrics / persistence                                                 #
    # ------------------------------------------------------------------ #

    def save_portfolio_values(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"date": self.date_memory, "portfolio_value": self.portfolio_values}
        )

    def get_portfolio_metrics(self) -> Dict:
        returns = np.array(self.portfolio_returns[1:])
        if len(returns) == 0:
            return {}

        total_return = (
            self.portfolio_value - self.initial_amount
        ) / self.initial_amount
        days = len(returns)
        annual_return = (
            (1 + total_return) ** (TRADING_DAYS / days) - 1 if days > 0 else 0.0
        )
        volatility = returns.std() * np.sqrt(TRADING_DAYS)

        risk_free_rate = 0.045
        sharpe = (
            (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        )

        neg = returns[returns < 0]
        downside_std = neg.std() * np.sqrt(TRADING_DAYS) if len(neg) > 0 else 0.0
        sortino = (
            (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
        )

        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = float(drawdown.max())

        sorted_ret = np.sort(returns)
        cvar_cut = max(1, int(0.05 * len(sorted_ret)))
        cvar = float(sorted_ret[:cvar_cut].mean())

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": -max_drawdown,
            "cvar_5": cvar,
        }


if __name__ == "__main__":
    print("Portfolio Environment module loaded successfully (gymnasium)")
