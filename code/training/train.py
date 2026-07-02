"""
Main training script for RL agents.

This script:
1. Loads and processes data
2. Creates training environment
3. Trains DRL agents (PPO, DDPG, SAC, QR-DDPG)
4. Evaluates agents
5. Saves results

"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")

# Resolve project root independent of CWD
_ROOT = Path(__file__).resolve().parent.parent  # → code/
sys.path.insert(0, str(_ROOT))  # add code/ so subpackages resolve

from agents import QRDDPGAgent
from data import DataProcessor
from environment import PortfolioEnv


class TrainDRLAgents:
    """Train and evaluate DRL agents for portfolio optimisation."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _ROOT / "config" / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path(self.config["output"]["results_dir"])
        self.models_dir = Path(self.config["output"]["models_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.train_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None

        print("Training configuration loaded from:", config_path)

    # ------------------------------------------------------------------ #
    # Data                                                                  #
    # ------------------------------------------------------------------ #

    def prepare_data(self) -> None:
        print("\n" + "=" * 50)
        print("STEP 1: Data Preparation")
        print("=" * 50)

        processor = DataProcessor(self.config)
        self.train_data, self.test_data = processor.process_all()

        print(f"\nTrain data shape : {self.train_data.shape}")
        print(f"Test  data shape : {self.test_data.shape}")
        print(f"Number of assets : {self.train_data['tic'].nunique()}")

    # ------------------------------------------------------------------ #
    # Environment factory                                                   #
    # ------------------------------------------------------------------ #

    def _make_env(self, data: pd.DataFrame) -> DummyVecEnv:
        """
        Wrap a PortfolioEnv in DummyVecEnv for SB3 compatibility.
        Uses a closure so each call creates a fresh environment.
        """
        env_cfg = self.config["environment"]
        risk_cfg = self.config["risk"]

        def _factory():
            return PortfolioEnv(
                df=data,
                initial_amount=env_cfg["initial_amount"],
                transaction_cost_pct=env_cfg["transaction_cost_pct"],
                max_drawdown_penalty=risk_cfg["max_drawdown_penalty"],
                hmax=env_cfg.get("hmax", 0.30),
                print_verbosity=env_cfg.get("print_verbosity", 50),
            )

        return DummyVecEnv([_factory])

    def _make_raw_env(self, data: pd.DataFrame) -> PortfolioEnv:
        """Create an unwrapped PortfolioEnv (for custom agents and evaluation)."""
        env_cfg = self.config["environment"]
        risk_cfg = self.config["risk"]
        return PortfolioEnv(
            df=data,
            initial_amount=env_cfg["initial_amount"],
            transaction_cost_pct=env_cfg["transaction_cost_pct"],
            max_drawdown_penalty=risk_cfg["max_drawdown_penalty"],
            hmax=env_cfg.get("hmax", 0.30),
            print_verbosity=env_cfg.get("print_verbosity", 1000),
        )

    # ------------------------------------------------------------------ #
    # SB3 agent training helpers                                            #
    # ------------------------------------------------------------------ #

    def _sb3_train(self, ModelClass, model_kwargs: dict, name: str, seed: int):
        env = self._make_env(self.train_data)
        train_cfg = self.config["training"]

        model = ModelClass(
            "MlpPolicy",
            env,
            **model_kwargs,
            policy_kwargs=dict(net_arch=[128, 64]),
            verbose=0,
            seed=seed,
        )
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            log_interval=train_cfg["log_interval"],
        )
        path = self.models_dir / f"{name}_seed_{seed}"
        model.save(str(path))
        return model

    def train_ppo(self, seed: int = 0):
        print(f"\nTraining PPO (seed={seed})...")
        cfg = self.config["models"]["ppo"]
        return self._sb3_train(
            PPO,
            dict(
                learning_rate=cfg["learning_rate"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                n_epochs=cfg["n_epochs"],
                gamma=0.99,
                gae_lambda=cfg["gae_lambda"],
                clip_range=cfg["clip_range"],
                ent_coef=cfg["ent_coef"],
                vf_coef=cfg["vf_coef"],
                max_grad_norm=cfg["max_grad_norm"],
            ),
            name="ppo",
            seed=seed,
        )

    def train_ddpg(self, seed: int = 0):
        print(f"\nTraining DDPG (seed={seed})...")
        cfg = self.config["models"]["ddpg"]
        return self._sb3_train(
            DDPG,
            dict(
                learning_rate=cfg["learning_rate_actor"],
                buffer_size=cfg["buffer_size"],
                batch_size=cfg["batch_size"],
                tau=cfg["tau"],
                gamma=cfg["gamma"],
            ),
            name="ddpg",
            seed=seed,
        )

    def train_sac(self, seed: int = 0):
        print(f"\nTraining SAC (seed={seed})...")
        cfg = self.config["models"]["sac"]
        return self._sb3_train(
            SAC,
            dict(
                learning_rate=cfg["learning_rate"],
                buffer_size=cfg["buffer_size"],
                batch_size=cfg["batch_size"],
                tau=cfg["tau"],
                gamma=cfg["gamma"],
                ent_coef=cfg["ent_coef"],
            ),
            name="sac",
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    # QR-DDPG training                                                      #
    # ------------------------------------------------------------------ #

    def train_qr_ddpg(self, seed: int = 0) -> QRDDPGAgent:
        print(f"\nTraining QR-DDPG (seed={seed})...")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Use the raw env: QRDDPGAgent is not SB3-compatible
        env = self._make_raw_env(self.train_data)
        cfg = self.config["models"]["qr_ddpg"]
        train_cfg = self.config["training"]

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        agent = QRDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=cfg["learning_rate_actor"],
            lr_critic=cfg["learning_rate_critic"],
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            n_quantiles=cfg["n_quantiles"],
            buffer_size=cfg["buffer_size"],
            device=device,
        )

        total_steps = train_cfg["total_timesteps"]
        batch_size = cfg["batch_size"]

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_count = 0

        for step in range(total_steps):
            action = agent.select_action(obs, noise=0.1)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            ep_reward += reward
            obs = next_obs

            if done:
                ep_count += 1
                if ep_count % 10 == 0:
                    print(f"  Episode {ep_count:4d} | Reward: {ep_reward:10.2f}")
                obs, _ = env.reset()
                ep_reward = 0.0

        # Save weights
        save_path = self.models_dir / f"qr_ddpg_seed_{seed}.pt"
        torch.save(
            {
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
            },
            save_path,
        )
        return agent

    # ------------------------------------------------------------------ #
    # Evaluation                                                            #
    # ------------------------------------------------------------------ #

    def evaluate_agent(self, model, agent_type: str = "sb3"):
        """
        Evaluate a trained agent on the held-out test data.

        Fix: both sb3 and custom agents now receive the *unwrapped*
        PortfolioEnv, preventing the shape mismatch that occurred when the
        original code passed DummyVecEnv output to QRDDPGAgent.select_action.
        """
        if self.test_data is None:
            raise RuntimeError(
                "test_data is not available: call prepare_data() before evaluate_agent()"
            )
        # Always use the raw (unwrapped) env for evaluation
        env = self._make_raw_env(self.test_data)
        obs, _ = env.reset()
        done = False

        while not done:
            if agent_type == "sb3":
                # model.predict expects (n_envs, obs_dim); add batch dim
                action, _ = model.predict(obs[np.newaxis], deterministic=True)
                action = action[0]  # remove batch dim
            else:
                action = model.select_action(obs, noise=0.0)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        metrics = env.get_portfolio_metrics()
        portfolio_df = env.save_portfolio_values()
        return metrics, portfolio_df

    # ------------------------------------------------------------------ #
    # Full training run                                                     #
    # ------------------------------------------------------------------ #

    def train_all_agents(self, n_seeds: int | None = None) -> dict:
        if n_seeds is None:
            n_seeds = self.config["training"]["n_seeds"]

        print("\n" + "=" * 50)
        print("STEP 2: Training DRL Agents")
        print("=" * 50)

        results: dict = {"ppo": [], "ddpg": [], "sac": [], "qr_ddpg": []}

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")

            ppo_model = self.train_ppo(seed)
            results["ppo"].append(self.evaluate_agent(ppo_model, "sb3")[0])

            ddpg_model = self.train_ddpg(seed)
            results["ddpg"].append(self.evaluate_agent(ddpg_model, "sb3")[0])

            sac_model = self.train_sac(seed)
            results["sac"].append(self.evaluate_agent(sac_model, "sb3")[0])

            qr_agent = self.train_qr_ddpg(seed)
            results["qr_ddpg"].append(self.evaluate_agent(qr_agent, "custom")[0])

            print(f"Seed {seed} completed")

        self._save_results(results)
        return results

    # ------------------------------------------------------------------ #
    # Persistence                                                           #
    # ------------------------------------------------------------------ #

    def _save_results(self, results: dict) -> None:
        rows = []
        for agent_name, metrics_list in results.items():
            for seed, metrics in enumerate(metrics_list):
                row = {"agent": agent_name, "seed": seed}
                row.update(metrics)
                rows.append(row)

        df = pd.DataFrame(rows)
        out = self.results_dir / "training_results.csv"
        df.to_csv(out, index=False)
        print(f"\nResults saved to: {out}")

        summary = df.groupby("agent").agg(
            {
                "annual_return": ["mean", "std"],
                "sharpe_ratio": ["mean", "std"],
                "max_drawdown": ["mean", "std"],
                "sortino_ratio": ["mean", "std"],
            }
        )
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(summary)


# ---------------------------------------------------------------------------


def main() -> None:
    trainer = TrainDRLAgents()
    trainer.prepare_data()
    trainer.train_all_agents(n_seeds=2)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
