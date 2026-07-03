"""
Reward Function Ablation Study Module.

Systematically varies the lambda (drawdown penalty) parameter
to analyze its impact on agent performance.

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

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


import torch
import yaml

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent  # → code/


class RewardAblationStudy:
    """Perform ablation study on the reward function's drawdown-penalty parameter."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _ROOT / "config" / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.lambda_values: List[float] = self.config["reward_ablation"][
            "lambda_values"
        ]
        self.n_seeds: int = self.config["reward_ablation"]["n_seeds"]
        self.output_dir: Path = Path(self.config["reward_ablation"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main study runner                                                     #
    # ------------------------------------------------------------------ #

    def run_ablation_study(
        self,
        agent_class,
        env_factory,
        training_steps: int = 100_000,
    ) -> pd.DataFrame:
        """
        Train *agent_class* for each lambda value and seed, then evaluate.

        Parameters
        ----------
        agent_class : callable
            Callable(env) → agent with .train(total_timesteps) and
            .predict(obs, deterministic) methods (SB3-compatible).
        env_factory : callable
            Callable(max_drawdown_penalty=float) → gymnasium.Env.
        training_steps : int
            Timesteps per configuration.
        """
        results: List[Dict] = []

        for lam in self.lambda_values:
            print(f"\n{'='*60}\nLambda = {lam}\n{'='*60}")

            for seed in range(self.n_seeds):
                print(f"  Seed {seed + 1}/{self.n_seeds}")
                np.random.seed(seed)
                torch.manual_seed(seed)

                env = env_factory(max_drawdown_penalty=lam)
                agent = agent_class(env)
                agent.learn(total_timesteps=training_steps)

                metrics = self._evaluate_agent(agent, env)
                metrics["lambda"] = lam
                metrics["seed"] = seed
                results.append(metrics)

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "ablation_results.csv", index=False)
        return df

    # ------------------------------------------------------------------ #
    # Agent evaluation (fixed)                                             #
    # ------------------------------------------------------------------ #

    def _evaluate_agent(self, agent, env) -> Dict:
        """
        Run n_eval_episodes evaluation episodes and aggregate metrics.

        Fix: get_portfolio_metrics() is called exactly once per episode,
        immediately after the terminal step, instead of being called
        multiple extra times on a stale environment state.
        """
        n_episodes = 10
        ep_returns: List[float] = []
        ep_sharpes: List[float] = []
        ep_drawdowns: List[float] = []
        ep_cvars: List[float] = []
        ep_vols: List[float] = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            # Capture metrics once, immediately after episode ends
            m = env.get_portfolio_metrics()
            ep_returns.append(m.get("annual_return", 0.0) * 100)
            ep_sharpes.append(m.get("sharpe_ratio", 0.0))
            ep_drawdowns.append(m.get("max_drawdown", 0.0))
            ep_cvars.append(m.get("cvar_5", 0.0))
            ep_vols.append(m.get("volatility", 0.0) * 100)

        return {
            "annual_return": float(np.mean(ep_returns)),
            "sharpe_ratio": float(np.mean(ep_sharpes)),
            "max_drawdown": float(np.mean(ep_drawdowns)),
            "cvar": float(np.mean(ep_cvars)),
            "volatility": float(np.mean(ep_vols)),
        }

    # ------------------------------------------------------------------ #
    # Analysis                                                              #
    # ------------------------------------------------------------------ #

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        grouped = results_df.groupby("lambda")

        stats = grouped.agg(
            {
                "sharpe_ratio": ["mean", "std"],
                "annual_return": ["mean", "std"],
                "max_drawdown": ["mean", "std"],
                "cvar": ["mean", "std"],
                "volatility": ["mean", "std"],
            }
        )

        mean_sharpe = grouped["sharpe_ratio"].mean()
        mean_drawdown = grouped["max_drawdown"].mean()
        mean_cvar = grouped["cvar"].mean()

        return {
            "statistics": stats,
            "optimal_lambda_sharpe": mean_sharpe.idxmax(),
            "optimal_lambda_drawdown": mean_drawdown.idxmax(),
            "optimal_lambda_cvar": mean_cvar.idxmax(),
            "sharpe_vs_drawdown": (
                results_df[["lambda", "sharpe_ratio", "max_drawdown"]]
                .groupby("lambda")
                .mean()
            ),
        }

    # ------------------------------------------------------------------ #
    # Plotting                                                              #
    # ------------------------------------------------------------------ #

    def plot_performance_surface(
        self, results_df: pd.DataFrame, save_path: str | None = None
    ) -> plt.Figure:
        agg = results_df.groupby("lambda").agg(
            {
                "sharpe_ratio": ["mean", "std"],
                "annual_return": ["mean", "std"],
                "max_drawdown": ["mean", "std"],
                "cvar": ["mean", "std"],
                "volatility": ["mean", "std"],
            }
        )
        lv = agg.index.values

        fig = plt.figure(figsize=(20, 12))

        def _plot(pos, key, label, color, vline_max=True):
            ax = plt.subplot(2, 3, pos)
            mu = agg[(key, "mean")].values
            std = agg[(key, "std")].values
            ax.plot(lv, mu, "o-", lw=2, ms=8, color=color)
            ax.fill_between(lv, mu - std, mu + std, alpha=0.3, color=color)
            ax.set(xlabel="Lambda", ylabel=label, title=f"{label} vs Lambda")
            ax.grid(True, alpha=0.3)
            if vline_max:
                ax.axvline(lv[np.argmax(mu)], color="red", ls="--", alpha=0.5)

        _plot(1, "sharpe_ratio", "Sharpe Ratio", "#1f77b4")
        _plot(2, "annual_return", "Annual Return (%)", "green")
        _plot(3, "max_drawdown", "Max Drawdown (%)", "red", vline_max=False)
        _plot(4, "cvar", "CVaR 5% (%)", "purple")
        _plot(5, "volatility", "Volatility (%)", "orange")

        # Heatmap
        ax6 = plt.subplot(2, 3, 6)
        sharpe = agg[("sharpe_ratio", "mean")].values
        ret = agg[("annual_return", "mean")].values
        dd = agg[("max_drawdown", "mean")].values

        def _norm(x):
            _require_seaborn()
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

        matrix = np.vstack([_norm(sharpe), _norm(ret), _norm(dd)])
        sns.heatmap(
            matrix,
            xticklabels=[f"{l:.1f}" for l in lv],
            yticklabels=["Sharpe", "Return", "Drawdown"],
            cmap="RdYlGn",
            ax=ax6,
            cbar_kws={"label": "Normalised"},
        )
        ax6.set(title="Performance Heatmap (Normalised)", xlabel="Lambda")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_tradeoff_frontier(
        self, results_df: pd.DataFrame, save_path: str | None = None
    ) -> plt.Figure:
        agg = results_df.groupby("lambda").agg(
            annual_return=("annual_return", "mean"),
            volatility=("volatility", "mean"),
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(
            agg["volatility"],
            agg["annual_return"],
            c=agg.index,
            s=200,
            cmap="viridis",
            edgecolors="black",
            lw=2,
            alpha=0.7,
        )
        for idx, row in agg.iterrows():
            ax.annotate(
                f"λ={idx:.1f}",
                (row["volatility"], row["annual_return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )
        plt.colorbar(sc, ax=ax, label="Lambda")
        ax.set(
            xlabel="Volatility (%)",
            ylabel="Annual Return (%)",
            title="Risk-Return Trade-off Frontier\n(Impact of Lambda Parameter)",
        )
        ax.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------ #
    # Report                                                                #
    # ------------------------------------------------------------------ #

    def generate_ablation_report(
        self,
        results_df: pd.DataFrame,
        analysis: Dict,
        output_path: str | None = None,
    ) -> str:
        lines = [
            "=" * 80,
            "REWARD FUNCTION ABLATION STUDY REPORT",
            "=" * 80,
            "",
            "STUDY CONFIGURATION",
            "-" * 80,
            f"Lambda values tested : {self.lambda_values}",
            f"Seeds per config     : {self.n_seeds}",
            f"Total experiments    : {len(results_df)}",
            "",
            "OPTIMAL LAMBDA VALUES",
            "-" * 80,
            f"Best Sharpe Ratio   : λ = {analysis['optimal_lambda_sharpe']:.2f}",
            f"Best Max Drawdown   : λ = {analysis['optimal_lambda_drawdown']:.2f}",
            f"Best CVaR           : λ = {analysis['optimal_lambda_cvar']:.2f}",
            "",
            "PERFORMANCE STATISTICS BY LAMBDA",
            "-" * 80,
            analysis["statistics"].to_string(),
            "",
            "KEY INSIGHTS",
            "-" * 80,
        ]
        sharpe_by_lam = results_df.groupby("lambda")["sharpe_ratio"].mean()
        best = sharpe_by_lam.idxmax()
        worst = sharpe_by_lam.idxmin()
        lines += [
            f"1. Sharpe ratio peaks at λ = {best:.2f}",
            f"2. Performance degrades at λ = {worst:.2f}",
            f"3. Recommended range: λ ∈ [{max(0, best - 0.2):.1f}, {min(1.0, best + 0.2):.1f}]",
        ]
        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report)
        return report


if __name__ == "__main__":
    print("Reward Ablation Study module loaded successfully")
