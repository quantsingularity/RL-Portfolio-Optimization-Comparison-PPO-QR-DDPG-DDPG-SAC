# Deep Reinforcement Learning for Portfolio Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18666986.svg)](https://doi.org/10.5281/zenodo.18666986)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](infrastructure/Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](code/production/api.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comparative study of four Deep Reinforcement Learning algorithms PPO, QR-DDPG, DDPG, and SAC applied to continuous portfolio optimization across a 25-asset universe. Integrates transaction cost analysis, market regime classification, and real-world trading constraints in a production-ready deployment framework.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [DRL Agents](#drl-agents)
- [Advanced Analysis](#advanced-analysis)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Production Deployment](#production-deployment)
- [License](#license)

---

## Overview

| Category               | Feature                   | Description                                                      |
| :--------------------- | :------------------------ | :--------------------------------------------------------------- |
| **DRL Core**           | Agent Comparison          | PPO, QR-DDPG, DDPG, SAC for continuous portfolio allocation      |
| **Financial Modeling** | Market Constraints        | Short-selling control, leverage limits, sector exposure caps     |
| **Advanced Analysis**  | Transaction Cost Analysis | Performance under retail and institutional cost structures       |
| **Advanced Analysis**  | Reward Ablation           | Risk-aversion parameter sweep across lambda in [0.0, 1.0]        |
| **Advanced Analysis**  | Market Regime Analysis    | Bull, Bear, Sideways classification via VIX and trend indicators |
| **Benchmarking**       | Extended Benchmarks       | 60/40, All-Weather, Risk-Parity, Minimum Correlation             |
| **Deployment**         | Production Stack          | FastAPI, PostgreSQL, Redis, Celery, Grafana via Docker Compose   |

---

## Quick Start

### Run script (simplest)

```bash
pip install -r code/requirements.txt  # install dependencies
bash scripts/run.sh test      # run the unit test suite (17 tests)
bash scripts/run.sh demo      # offline smoke run on synthetic data (~2 min, no network)
bash scripts/run.sh data      # data pipeline only
bash scripts/run.sh train     # train all agents (PPO, DDPG, SAC, QR-DDPG)
bash scripts/run.sh evaluate  # evaluate agents vs classical benchmarks
bash scripts/run.sh api       # launch the FastAPI production service
```

Offline environments are fully supported: set `data.use_synthetic_data: true`
in `code/config/config.yaml` (or rely on the automatic fallback when Yahoo
Finance is unreachable) to run the entire pipeline on reproducible synthetic
data.

### Docker (Recommended)

```bash
git clone https://github.com/quantsingularity/RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC.git
cd RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC

docker-compose up -d
```

| URL                          | Service           |
| :--------------------------- | :---------------- |
| `http://localhost:8000/docs` | API Documentation |
| `http://localhost:8888`      | Jupyter Notebooks |
| `http://localhost:3000`      | Grafana Dashboard |

### Local Installation

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r code/requirements.txt -r code/requirements-prod.txt

python code/data/data_processor.py
python code/training/train.py
python code/evaluation/evaluate.py
```

---

## Project Structure

| Path                 | Description           | Key Files                                                         |
| :------------------- | :-------------------- | :---------------------------------------------------------------- |
| `code/`              | Core ML logic         | `agents.py`, `environment.py`, `train.py`, `evaluate.py`          |
| `code/config/`       | Configuration         | `config.yaml`                                                     |
| `code/production/`   | Production API        | `api.py`                                                          |
| `code/tests/`        | Unit tests            | `test_environment.py`, `test_agents.py`, `test_data_processor.py` |
| `infrastructure/`    | Infra configs         | `.env.example`                                                    |
| `Dockerfile`         | Container build       | Multi-stage, CUDA-enabled image                                   |
| `docker-compose.yml` | Service orchestration | All services (API, DB, Redis, Jupyter, Grafana)                   |
| `scripts/`           | Dev & CI helpers      | `run.sh`, `Makefile`, `lint.sh`                                   |
| `docs/`              | Documentation         | `QUICKSTART.md`                                                   |

---

## Configuration

All settings are managed via `config/config.yaml`.

| Group               | Key Parameters                                                      | Defaults                                           |
| :------------------ | :------------------------------------------------------------------ | :------------------------------------------------- |
| `data`              | 25 assets (4 classes), `start_date`, technical indicators           | `start_date: 2015-01-01`, `test_start: 2023-01-01` |
| `environment`       | `initial_amount`, `transaction_cost_pct`, `slippage_coefficient`    | `initial_amount: 1,000,000`, `tc: 0.001`           |
| `constraints`       | `short_selling`, `max_leverage`, `max_position_size`, sector limits | `max_leverage: 1.0`, `equities: 0.6`               |
| `transaction_costs` | Cost structures (retail, institutional), rebalance frequencies      | `retail: 0.005`, `weekly: 5`                       |
| `regime_analysis`   | Classification methods, VIX and return thresholds                   | `bull: vix < 20`, `bear: vix > 30`                 |
| `training`          | `total_timesteps`, `n_eval_episodes`, `n_seeds`                     | `timesteps: 100,000`, `seeds: 10`                  |
| `production`        | API host/port, rebalancing schedule, risk monitoring                | `port: 8000`, `rebalancing: weekly`                |

---

## DRL Agents

### Algorithm Comparison

| Algorithm   | Type            | Core Concept                                              | Key Advantage                         |
| :---------- | :-------------- | :-------------------------------------------------------- | :------------------------------------ |
| **PPO**     | Policy Gradient | Clipped surrogate objective with minibatch updates        | Stability and sample efficiency       |
| **QR-DDPG** | Actor-Critic    | Models Q-function distribution via quantile regression    | Risk-awareness and tail risk control  |
| **DDPG**    | Actor-Critic    | Deterministic policy with Q-function in continuous spaces | Simplicity and determinism            |
| **SAC**     | Max Entropy RL  | Entropy term in reward encourages exploration             | High sample efficiency and robustness |

### Hyperparameters

| Parameter         | PPO            | QR-DDPG         | DDPG            | SAC       |
| :---------------- | :------------- | :-------------- | :-------------- | :-------- |
| `total_timesteps` | 100,000        | 100,000         | 100,000         | 100,000   |
| `batch_size`      | 256            | 128             | 128             | 256       |
| `learning_rate`   | 0.0003         | 0.0001 / 0.0003 | 0.0001 / 0.0003 | 0.0003    |
| `buffer_size`     | N/A            | 1,000,000       | 1,000,000       | 1,000,000 |
| `ent_coef`        | 0.01           | N/A             | N/A             | 0.2       |
| `policy_kwargs`   | [128, 64] ReLU | [128, 64]       | [128, 64]       | [128, 64] |

---

## Advanced Analysis

### 1. Transaction Cost Analysis

Evaluates trading cost impact and optimal rebalancing frequency.

- Retail costs (0.5%) reduce annual returns by 15-20%
- Optimal rebalancing frequency: weekly to biweekly for most DRL strategies

```python
from code.transaction_cost_analysis import TransactionCostAnalyzer

analyzer = TransactionCostAnalyzer()
results = analyzer.analyze_rebalancing_frequency(
    strategy_name='ppo',
    portfolio_values_base=portfolio_values,
    portfolio_weights_history=weights_history,
    dates=dates
)
analyzer.plot_cost_impact(results, save_path='results/cost_analysis.png')
```

### 2. Reward Ablation Study

Sweeps the risk-aversion parameter lambda in the reward function to find the optimal risk/return balance.

- Optimal lambda = 0.5 delivers the best Sharpe/MaxDrawdown tradeoff

```python
from code.reward_ablation import RewardAblationStudy

study = RewardAblationStudy()
results = study.run_ablation_study(
    agent_class=PPO,
    env_factory=create_env_with_lambda,
    training_steps=100000
)
study.plot_performance_surface(results, 'results/ablation_surface.png')
```

### 3. Market Regime Analysis

Classifies market conditions using VIX and trend indicators, then evaluates agent performance per regime.

- **Bull markets:** SAC leads (Sharpe 2.5+) via superior exploration
- **Bear markets:** QR-DDPG leads (lowest CVaR) via quantile risk modeling
- **Sideways markets:** PPO delivers most consistent, stable performance

```python
from code.regime_analysis import MarketRegimeAnalyzer

analyzer = MarketRegimeAnalyzer()
regime_df = analyzer.identify_regimes_vix(market_data)
performance = analyzer.analyze_performance_by_regime(
    strategy_results={'ppo': ppo_results, 'qr_ddpg': qr_results},
    regime_labels=regime_df
)
```

---

## Benchmarks

| Category     | Strategy              | Description                             |
| :----------- | :-------------------- | :-------------------------------------- |
| Traditional  | Equal Weight          | Equal allocation across all assets      |
| Traditional  | 60/40 Portfolio       | 60% equities, 40% bonds                 |
| MPT          | Minimum Volatility    | Minimizes historical portfolio variance |
| MPT          | Mean-Variance (MVO)   | Maximizes Sharpe for a given risk level |
| Risk-Based   | Risk-Parity           | Equal risk contribution per asset       |
| Risk-Based   | Minimum Correlation   | Minimizes intra-portfolio correlation   |
| Factor-Based | Momentum              | Invests in recent top performers        |
| Advanced     | All-Weather Portfolio | Ray Dalio's all-environment strategy    |

---

## Results

### Performance Summary (Test Period: 2023-2024)

| Strategy        | Sharpe   | Annual Return | Max Drawdown | CVaR 5%   | Cost Impact (0.1% TC) |
| :-------------- | :------- | :------------ | :----------- | :-------- | :-------------------- |
| **PPO**         | **2.15** | **38.2%**     | -7.2%        | -1.8%     | -2.1%                 |
| **QR-DDPG**     | 2.08     | 36.5%         | **-6.5%**    | **-1.5%** | -1.8%                 |
| **SAC**         | 1.98     | 35.1%         | -8.8%        | -2.1%     | -2.3%                 |
| **DDPG**        | 1.85     | 32.1%         | -9.5%        | -2.5%     | -2.0%                 |
| 60/40 Portfolio | 1.52     | 22.5%         | -11.2%       | -3.2%     | -0.8%                 |
| All-Weather     | 1.65     | 24.1%         | -9.5%        | -2.8%     | -0.6%                 |
| Risk-Parity     | 1.45     | 25.8%         | -12.1%       | -3.1%     | -1.2%                 |

PPO and QR-DDPG outperform all traditional benchmarks on risk-adjusted returns, drawdown, and tail risk.

### Performance by Market Regime

| Algorithm   | Bull Sharpe | Bear Sharpe | Sideways Sharpe | Strength                      |
| :---------- | :---------- | :---------- | :-------------- | :---------------------------- |
| **PPO**     | 2.3         | 1.8         | **2.0**         | Consistent across all regimes |
| **QR-DDPG** | 2.1         | **2.2**     | 1.9             | Bear market risk control      |
| **SAC**     | **2.5**     | 1.5         | 1.8             | Bull market exploration       |
| **DDPG**    | 1.9         | 1.2         | 1.5             | Baseline reference            |

---

### Portfolio Recommendation Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/portfolio/recommend" \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client_001", "risk_tolerance": "medium", "investment_amount": 1000000}'
```

```json
{
  "client_id": "client_001",
  "timestamp": "2024-01-15T10:30:00",
  "weights": { "AAPL": 0.12, "MSFT": 0.1, "BTC-USD": 0.08 },
  "expected_return": 25.3,
  "expected_volatility": 18.5,
  "sharpe_ratio": 1.85
}
```

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
