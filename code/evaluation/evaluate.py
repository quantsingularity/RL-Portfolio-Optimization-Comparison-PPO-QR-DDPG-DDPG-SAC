"""
Evaluation script for comparing all strategies (DRL + Benchmarks).

This script:
1. Evaluates trained DRL agents
2. Backtests benchmark strategies
3. Performs statistical significance testing
4. Generates comparison tables and figures

"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import f_oneway
from stable_baselines3 import DDPG, PPO, SAC
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from utils import console as ui

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent  # → code/
sys.path.insert(0, str(_ROOT))  # add code/ so subpackages resolve

from agents import QRDDPGAgent
from data import DataProcessor
from environment import PortfolioEnv
from evaluation.benchmark_strategies import (
    BacktestBenchmark,
)  # direct import avoids circular


class EvaluateStrategies:
    """Evaluate and compare all portfolio strategies."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _ROOT / "config" / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path(self.config["output"]["results_dir"])
        self.models_dir = Path(self.config["output"]["models_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.test_data: pd.DataFrame | None = None
        print("Evaluation configuration loaded from:", config_path)

    # ------------------------------------------------------------------ #
    # Data                                                                  #
    # ------------------------------------------------------------------ #

    def load_data(self) -> None:
        processor = DataProcessor(self.config)
        _, self.test_data = processor.process_all()
        print(f"Test data loaded: {self.test_data.shape}")

    # ------------------------------------------------------------------ #
    # Environment factory (raw, unwrapped)                                  #
    # ------------------------------------------------------------------ #

    def _make_env(self) -> PortfolioEnv:
        env_cfg = self.config["environment"]
        risk_cfg = self.config["risk"]
        return PortfolioEnv(
            df=self.test_data,
            initial_amount=env_cfg["initial_amount"],
            transaction_cost_pct=env_cfg["transaction_cost_pct"],
            max_drawdown_penalty=risk_cfg["max_drawdown_penalty"],
            hmax=env_cfg.get("hmax", 0.30),
            print_verbosity=1000,
        )

    # ------------------------------------------------------------------ #
    # DRL agents                                                            #
    # ------------------------------------------------------------------ #

    def evaluate_drl_agents(self) -> pd.DataFrame:
        ui.section("A", "Evaluating DRL Agents")

        results = []
        n_seeds = self.config["training"]["n_seeds"]

        for agent_name, ModelClass, file_ext in [
            ("PPO", PPO, ".zip"),
            ("DDPG", DDPG, ".zip"),
            ("SAC", SAC, ".zip"),
        ]:
            print(f"\nEvaluating {agent_name}...")
            for seed in range(n_seeds):
                path = self.models_dir / f"{agent_name.lower()}_seed_{seed}"
                if (path.parent / (path.name + file_ext)).exists():
                    model = ModelClass.load(str(path))
                    metrics, portfolio_df = self._evaluate_sb3(model, agent_name, seed)
                    results.append(metrics)
                    if seed == 0:
                        portfolio_df.to_csv(
                            self.results_dir
                            / f"{agent_name.lower()}_portfolio_values.csv",
                            index=False,
                        )

        print("\nEvaluating QR-DDPG...")
        for seed in range(n_seeds):
            path = self.models_dir / f"qr_ddpg_seed_{seed}.pt"
            if path.exists():
                metrics, portfolio_df = self._evaluate_qr_ddpg(str(path), seed)
                results.append(metrics)
                if seed == 0:
                    portfolio_df.to_csv(
                        self.results_dir / "qr_ddpg_portfolio_values.csv",
                        index=False,
                    )

        drl_df = pd.DataFrame(results)
        drl_df.to_csv(self.results_dir / "drl_evaluation_results.csv", index=False)
        return drl_df

    def _evaluate_sb3(self, model, agent_name: str, seed: int):
        """Run one evaluation episode with an SB3 model."""

        env = self._make_env()
        obs, _ = env.reset()
        done = False

        while not done:
            # model.predict expects a batched observation
            action, _ = model.predict(obs[np.newaxis], deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated

        metrics = env.get_portfolio_metrics()
        metrics["agent"] = agent_name
        metrics["seed"] = seed
        return metrics, env.save_portfolio_values()

    def _evaluate_qr_ddpg(self, model_path: str, seed: int):
        """Load and run one evaluation episode with a QR-DDPG agent."""
        env = self._make_env()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = QRDDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])

        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs, noise=0.0)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        metrics = env.get_portfolio_metrics()
        metrics["agent"] = "QR-DDPG"
        metrics["seed"] = seed
        return metrics, env.save_portfolio_values()

    # ------------------------------------------------------------------ #
    # Benchmark strategies                                                  #
    # ------------------------------------------------------------------ #

    def evaluate_benchmarks(self) -> pd.DataFrame:
        ui.section("B", "Evaluating Benchmark Strategies")

        # Build asset_classes mapping from config
        asset_classes: dict = {}
        for cls, tickers in self.config["data"]["assets"].items():
            for t in tickers:
                asset_classes[t] = cls

        backtester = BacktestBenchmark(
            df=self.test_data,
            initial_amount=self.config["environment"]["initial_amount"],
            transaction_cost_pct=self.config["environment"]["transaction_cost_pct"],
            rebalance_freq=20,
            asset_classes=asset_classes,
        )

        strategies = self.config.get(
            "benchmarks",
            [
                "equal_weight",
                "mvo",
                "risk_parity",
                "minimum_volatility",
                "momentum",
                "sixty_forty",
                "all_weather",
                "minimum_correlation",
            ],
        )

        results = []
        for strategy in strategies:
            print(f"  Backtesting {strategy}...")
            try:
                result = backtester.backtest_strategy(strategy)
            except Exception as exc:
                print(f"  [WARN] {strategy} failed: {exc}")
                continue

            # Normalise to decimals so benchmark and DRL metrics are on the same scale.
            # DRL get_portfolio_metrics() returns decimals (e.g. 0.15 for 15%);
            # BacktestBenchmark._compute_metrics returns *_pct keys (e.g. 15.0).
            metrics = {
                "agent": strategy.upper(),
                "seed": 0,
                "annual_return": result["annual_return_pct"] / 100,
                "sharpe_ratio": result["sharpe_ratio"],
                "sortino_ratio": result["sortino_ratio"],
                "max_drawdown": result["max_drawdown_pct"] / 100,
                "cvar_5": result["cvar_5_pct"] / 100,
                "volatility": result["volatility_pct"] / 100,
            }
            results.append(metrics)

            pd.DataFrame(
                {"date": result["dates"], "portfolio_value": result["portfolio_values"]}
            ).to_csv(self.results_dir / f"{strategy}_portfolio_values.csv", index=False)

        benchmark_df = pd.DataFrame(results)
        benchmark_df.to_csv(
            self.results_dir / "benchmark_evaluation_results.csv", index=False
        )
        return benchmark_df

    # ------------------------------------------------------------------ #
    # Statistical testing                                                   #
    # ------------------------------------------------------------------ #

    def statistical_significance_test(
        self, drl_df: pd.DataFrame, benchmark_df: pd.DataFrame
    ) -> None:
        ui.section("C", "Statistical Significance Testing")

        all_results = pd.concat([drl_df, benchmark_df], ignore_index=True)

        groups = [
            grp["annual_return"].dropna().values
            for _, grp in all_results.groupby("agent")
            if grp["annual_return"].dropna().shape[0] > 0
        ]

        # One-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        ui.kv("ANOVA", f"F={f_stat:.2f}  p={p_value:.6f}")
        print(
            "  "
            + ui.badge(
                p_value < 0.05,
                "significant difference between strategies " "(p < 0.05)",
                "no significant difference between " "strategies (p >= 0.05)",
            )
        )

        # Tukey's HSD
        print("\nTukey HSD pairwise comparison:")
        tukey_data, tukey_labels = [], []
        for name, grp in all_results.groupby("agent"):
            vals = grp["annual_return"].dropna().values
            tukey_data.extend(vals)
            tukey_labels.extend([name] * len(vals))

        tukey_result = pairwise_tukeyhsd(tukey_data, tukey_labels, alpha=0.05)
        print(tukey_result)

        pd.DataFrame(
            data=tukey_result.summary().data[1:],
            columns=tukey_result.summary().data[0],
        ).to_csv(self.results_dir / "tukey_hsd_results.csv", index=False)

    # ------------------------------------------------------------------ #
    # Comparison table                                                      #
    # ------------------------------------------------------------------ #

    def create_comparison_table(
        self, drl_df: pd.DataFrame, benchmark_df: pd.DataFrame
    ) -> pd.DataFrame:
        ui.section("D", "Comparison Table")

        all_results = pd.concat([drl_df, benchmark_df], ignore_index=True)

        metrics = [
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "cvar_5",
        ]
        summary = all_results.groupby("agent")[metrics].agg(["mean", "std"]).round(4)

        rows = []
        for agent in summary.index:
            row = {"Strategy": agent}
            for m in metrics:
                mu = summary.loc[agent, (m, "mean")]
                std = summary.loc[agent, (m, "std")]
                row[m] = f"{mu:.2f} ± {std:.2f}" if not pd.isna(std) else f"{mu:.2f}"
            rows.append(row)

        col_map = {
            "annual_return": "Annual Return (%)",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "max_drawdown": "Max Drawdown (%)",
            "cvar_5": "CVaR 5% (%)",
        }
        cmp_df = pd.DataFrame(rows).rename(columns=col_map)
        cmp_df.to_csv(self.results_dir / "comparison_table.csv", index=False)

        print(cmp_df.to_string(index=False))
        return cmp_df

    # ------------------------------------------------------------------ #
    # Orchestrator                                                          #
    # ------------------------------------------------------------------ #

    def run_full_evaluation(self):
        self.load_data()
        drl_df = self.evaluate_drl_agents()
        benchmark_df = self.evaluate_benchmarks()
        self.statistical_significance_test(drl_df, benchmark_df)
        comparison_df = self.create_comparison_table(drl_df, benchmark_df)

        comparison_df.iloc[0] if len(comparison_df) else None
        ui.summary_panel(
            "EVALUATION COMPLETE",
            {
                "DRL agents evaluated": drl_df["agent"].nunique(),
                "Benchmarks": benchmark_df["agent"].nunique(),
                "Comparison rows": len(comparison_df),
            },
            footer=f"Tables and figures: {self.results_dir}/",
        )
        return drl_df, benchmark_df, comparison_df


def main() -> None:
    evaluator = EvaluateStrategies()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
