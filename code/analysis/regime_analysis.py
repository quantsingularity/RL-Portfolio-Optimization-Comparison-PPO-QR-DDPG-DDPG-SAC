"""
Market Regime Analysis Module.

Classifies market conditions and analyzes algorithm performance
across different regimes (bull, bear, sideways).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SEABORN = True
except Exception:  # missing seaborn, or seaborn too old for this matplotlib
    sns = None
    _HAS_SEABORN = False


def _require_seaborn():
    """Raise a clear error when a seaborn-based plot is requested."""
    if not _HAS_SEABORN:
        raise ImportError(
            "seaborn (>=0.13) is required for this plot but could not be "
            "imported. Older seaborn versions are incompatible with "
            "matplotlib >= 3.9 (register_cmap removal). "
            "Fix with: pip install -U 'seaborn>=0.13'"
        )


import yaml

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent  # → code/


class MarketRegimeAnalyzer:
    """Classify market conditions and analyse strategy performance per regime."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _ROOT / "config" / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.regime_cfg = self.config["regime_analysis"]
        self.output_dir = Path(self.regime_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Regime identification                                                 #
    # ------------------------------------------------------------------ #

    def identify_regimes_vix(
        self,
        market_data: pd.DataFrame,
        vix_column: str = "VIX",
    ) -> pd.DataFrame:
        """Label each row with a regime based on the VIX level."""
        df = market_data.copy()
        bull_thr = self.regime_cfg["regime_definitions"]["bull"]["vix_threshold"]
        bear_thr = self.regime_cfg["regime_definitions"]["bear"]["vix_threshold"]

        def _label(vix):
            if vix < bull_thr:
                return "bull"
            if vix > bear_thr:
                return "bear"
            return "sideways"

        df["regime"] = df[vix_column].apply(_label)
        return df

    def identify_regimes_trend(
        self,
        market_data: pd.DataFrame,
        price_column: str = "Close",
        short_window: int = 50,
        long_window: int = 200,
    ) -> pd.DataFrame:
        """Label regimes using a short/long moving-average crossover."""
        df = market_data.copy()
        df["SMA_short"] = df[price_column].rolling(short_window).mean()
        df["SMA_long"] = df[price_column].rolling(long_window).mean()

        def _label(row):
            if pd.isna(row["SMA_short"]) or pd.isna(row["SMA_long"]):
                return "sideways"
            diff = (row["SMA_short"] - row["SMA_long"]) / row["SMA_long"]
            if diff > 0.05:
                return "bull"
            if diff < -0.05:
                return "bear"
            return "sideways"

        df["regime"] = df.apply(_label, axis=1)
        return df

    def identify_regimes_returns(
        self,
        market_data: pd.DataFrame,
        returns_column: str = "returns",
        lookback_window: int = 60,
    ) -> pd.DataFrame:
        """Label regimes by comparing the rolling annualised return to thresholds."""
        df = market_data.copy()
        bull_ret = self.regime_cfg["regime_definitions"]["bull"]["return_threshold"]
        bear_ret = self.regime_cfg["regime_definitions"]["bear"]["return_threshold"]

        df["rolling_return"] = df[returns_column].rolling(lookback_window).mean() * 252

        def _label(ret):
            if pd.isna(ret):
                return "sideways"
            if ret > bull_ret:
                return "bull"
            if ret < bear_ret:
                return "bear"
            return "sideways"

        df["regime"] = df["rolling_return"].apply(_label)
        return df

    # ------------------------------------------------------------------ #
    # Performance by regime                                                 #
    # ------------------------------------------------------------------ #

    def analyze_performance_by_regime(
        self,
        strategy_results: Dict[str, pd.DataFrame],
        regime_labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cross-join strategy results with regime labels and compute per-regime metrics.

        Parameters
        ----------
        strategy_results : dict[str, pd.DataFrame]
            Keys are strategy names; values are DataFrames with at least
            'date', 'portfolio_value', and optionally 'returns' columns.
        regime_labels : pd.DataFrame
            Must have 'date' and 'regime' columns.
        """
        rows: list = []

        for name, res in strategy_results.items():
            merged = res.merge(
                regime_labels[["date", "regime"]], on="date", how="inner"
            )

            for regime in ["bull", "bear", "sideways"]:
                sub = merged[merged["regime"] == regime]
                if len(sub) == 0:
                    continue
                m = self._calculate_regime_metrics(sub)
                m["strategy"] = name
                m["regime"] = regime
                m["n_periods"] = len(sub)
                rows.append(m)

        return pd.DataFrame(rows)

    def compare_algorithms_by_regime(
        self, performance_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Return the best-performing strategy (by Sharpe) for each regime."""
        best: Dict[str, Dict] = {}
        for regime in ["bull", "bear", "sideways"]:
            sub = performance_df[performance_df["regime"] == regime]
            if sub.empty:
                continue
            idx = sub["sharpe_ratio"].idxmax()
            row = sub.loc[idx]
            best[regime] = {
                "strategy": row["strategy"],
                "sharpe_ratio": row["sharpe_ratio"],
                "annual_return": row["annual_return"],
                "max_drawdown": row["max_drawdown"],
            }
        return best

    # ------------------------------------------------------------------ #
    # Plots                                                                 #
    # ------------------------------------------------------------------ #

    def plot_regime_performance(
        self,
        performance_df: pd.DataFrame,
        save_path: str | None = None,
    ) -> plt.Figure:
        _require_seaborn()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for ax, metric, title in [
            (axes[0, 0], "sharpe_ratio", "Sharpe Ratio by Market Regime"),
            (axes[0, 1], "annual_return", "Annual Return (%) by Market Regime"),
            (axes[1, 0], "max_drawdown", "Max Drawdown (%) by Market Regime"),
        ]:
            pivot = performance_df.pivot_table(
                values=metric, index="strategy", columns="regime", aggfunc="mean"
            )
            pivot.plot(kind="bar", ax=ax, width=0.8)
            ax.set(title=title, xlabel="Strategy", ylabel=metric)
            ax.legend(title="Regime")
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Heatmap
        pivot_sr = performance_df.pivot_table(
            values="sharpe_ratio", index="strategy", columns="regime", aggfunc="mean"
        )
        sns.heatmap(
            pivot_sr,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.0,
            ax=axes[1, 1],
            cbar_kws={"label": "Sharpe Ratio"},
        )
        axes[1, 1].set(title="Sharpe Ratio Heatmap", xlabel="Regime", ylabel="Strategy")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_regime_timeline(
        self,
        regime_df: pd.DataFrame,
        price_data: pd.DataFrame | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        colours = {"bull": "green", "bear": "red", "sideways": "gray"}
        nrows = 2 if price_data is not None else 1
        fig, axes = plt.subplots(
            nrows, 1, figsize=(16, 10 if nrows == 2 else 5), sharex=True
        )
        if nrows == 1:
            axes = [axes]

        ax = axes[0]
        for regime, col in colours.items():
            sub = regime_df[regime_df["regime"] == regime]
            ax.scatter(
                sub["date"],
                [regime] * len(sub),
                c=col,
                s=10,
                alpha=0.6,
                label=regime.capitalize(),
            )
        ax.set(ylabel="Market Regime", title="Market Regime Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if price_data is not None:
            ax2 = axes[1]
            ax2.plot(price_data["date"], price_data["Close"], lw=1.5, color="black")
            for regime, col in colours.items():
                for d in regime_df[regime_df["regime"] == regime]["date"]:
                    ax2.axvspan(d, d, alpha=0.1, color=col)
            ax2.set(ylabel="Price", xlabel="Date", title="Price with Regime Overlay")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------ #
    # Report                                                                #
    # ------------------------------------------------------------------ #

    def generate_regime_report(
        self,
        performance_df: pd.DataFrame,
        best_performers: Dict,
        output_path: str | None = None,
    ) -> str:
        lines = [
            "=" * 80,
            "MARKET REGIME ANALYSIS REPORT",
            "=" * 80,
            "",
            "REGIME DISTRIBUTION",
            "-" * 80,
        ]
        counts = performance_df.groupby("regime")["n_periods"].first()
        total = counts.sum()
        for r, n in counts.items():
            lines.append(f"  {r.capitalize()}: {n} periods ({n/total*100:.1f}%)")

        lines += ["", "BEST PERFORMERS BY REGIME", "-" * 80]
        for regime, p in best_performers.items():
            lines += [
                f"\n{regime.upper()} MARKET:",
                f"  Strategy     : {p['strategy']}",
                f"  Sharpe Ratio : {p['sharpe_ratio']:.3f}",
                f"  Annual Return: {p['annual_return']:.2f}%",
                f"  Max Drawdown : {p['max_drawdown']:.2f}%",
            ]

        lines += ["", "DETAILED PERFORMANCE", "-" * 80]
        summary = performance_df.pivot_table(
            values=["sharpe_ratio", "annual_return", "max_drawdown"],
            index="strategy",
            columns="regime",
            aggfunc="mean",
        )
        lines.append(summary.to_string())

        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report)
        return report

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_regime_metrics(self, regime_data: pd.DataFrame) -> Dict:
        if "returns" in regime_data.columns:
            returns = regime_data["returns"].dropna().values
        else:
            returns = np.array([])

        if len(returns) == 0:
            return {
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
            }

        ann_ret = float(np.mean(returns) * 252 * 100)
        vol = float(np.std(returns) * np.sqrt(252) * 100)
        sharpe = (
            (np.mean(returns) * 252 - 0.045) / (np.std(returns) * np.sqrt(252))
            if np.std(returns) > 0
            else 0.0
        )

        if "portfolio_value" in regime_data.columns:
            vals = regime_data["portfolio_value"].values
            peak = np.maximum.accumulate(vals)
            max_dd = float(-np.max((peak - vals) / np.where(peak > 0, peak, 1)) * 100)
        else:
            max_dd = 0.0

        return {
            "annual_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "volatility": vol,
        }


if __name__ == "__main__":
    print("Market Regime Analyzer module loaded successfully")
