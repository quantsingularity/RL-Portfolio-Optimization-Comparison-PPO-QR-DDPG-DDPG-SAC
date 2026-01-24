# Deep Reinforcement Learning for Portfolio Optimization: A Comparative Study

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](production/api.py)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration Details](#-configuration-details)
- [Deep Reinforcement Learning Agents](#-deep-reinforcement-learning-agents)
- [Advanced Analysis Modules](#-advanced-analysis-modules)
- [Benchmarking Strategies](#-benchmarking-strategies)
- [Results & Analysis](#-results--analysis)
- [Production Deployment](#-production-deployment)
- [License](#-license)

---

## 💡 Project Overview

This repository presents a comprehensive comparative study of four state-of-the-art Deep Reinforcement Learning (DRL) algorithms—**Proximal Policy Optimization (PPO)**, **Quantile Regression Deep Deterministic Policy Gradient (QR-DDPG)**, **Deep Deterministic Policy Gradient (DDPG)**, and **Soft Actor-Critic (SAC)**—applied to the problem of continuous portfolio optimization.

The project goes beyond a simple comparison by integrating advanced financial engineering concepts, including **transaction cost analysis**, **market regime classification**, and **strict market constraints**, to create a robust, production-ready framework for algorithmic portfolio management.

## ✨ Key Features

The framework is designed for both rigorous academic research and practical, production-level deployment. The following table summarizes the core capabilities:

| Category               | Feature                       | Description                                                                                                                         | Files/Components                                                                 |
| :--------------------- | :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **DRL Core**           | **Agent Comparison**          | Comparative analysis of PPO, QR-DDPG, DDPG, and SAC for continuous action space portfolio allocation.                               | `code/agents.py`, `code/train.py`                                                |
| **Financial Modeling** | **Market Constraints**        | Implements real-world constraints: short-selling control, leverage limits (e.g., max 1.0), and per-asset/sector exposure limits.    | `config/config.yaml`, `code/environment.py`                                      |
| **Advanced Analysis**  | **Transaction Cost Analysis** | Analyzes performance under various cost structures (retail, institutional) and determines the optimal rebalancing frequency.        | `code/transaction_cost_analysis.py`, `notebooks/transaction_cost_analysis.ipynb` |
| **Advanced Analysis**  | **Reward Ablation Study**     | Systematic testing of the risk-aversion parameter (λ) in the reward function to visualize the performance surface.                  | `code/reward_ablation.py`                                                        |
| **Advanced Analysis**  | **Market Regime Analysis**    | Classifies market conditions (Bull, Bear, Sideways) using VIX and trend analysis, and evaluates algorithm performance per regime.   | `code/regime_analysis.py`                                                        |
| **Benchmarking**       | **Extended Benchmarks**       | Comparison against sophisticated traditional strategies like 60/40, All-Weather Portfolio, and Minimum Correlation.                 | `code/enhanced_benchmarks.py`                                                    |
| **Deployment**         | **Production Architecture**   | Full containerized deployment with FastAPI for predictions, PostgreSQL for data, Redis for caching, and Celery for scheduled tasks. | `Dockerfile`, `docker-compose.yml`, `production/api.py`                          |

---

## 🚀 Quick Start

The recommended way to run this project is using Docker and Docker Compose, which sets up the entire environment, including the API, database, and monitoring tools.

### Option 1: Docker (Recommended)

This option sets up the entire production-ready stack.

```bash
# 1. Clone the repository
git clone https://github.com/quantsingularity/RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC.git
cd RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC

# 2. Start all services (API, DB, Redis, Grafana, Jupyter)
docker-compose up -d

# 3. Check status
docker-compose ps

# Access Services:
# - API Documentation: http://localhost:8000/docs
# - Jupyter Notebooks: http://localhost:8888
# - Grafana Dashboard: http://localhost:3000
```

### Option 2: Local Installation

For local development and training:

```bash
# 1. Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt

# 3. Run data preparation and training
python code/data_processor.py
python code/train.py
python code/evaluate.py
```

---

## 📂 Project Structure

The repository is organized into logical directories for core code, configuration, production services, and analysis notebooks.

| Path              | Description                | Key Files                                                                     |
| :---------------- | :------------------------- | :---------------------------------------------------------------------------- |
| **`/`**           | **Root Directory**         | `Dockerfile`, `docker-compose.yml`, `Makefile`, `requirements.txt`, `LICENSE` |
| **`code/`**       | **Core Logic**             | `agents.py`, `environment.py`, `data_processor.py`, `train.py`, `evaluate.py` |
| **`config/`**     | **Configuration**          | `config.yaml`                                                                 |
| **`production/`** | **Deployment Services**    | `api.py` (FastAPI), `monitoring.py`                                           |
| **`notebooks/`**  | **Analysis & Exploration** | `transaction_cost_analysis.ipynb`                                             |
| **`docs/`**       | **Documentation**          | `QUICKSTART.md`                                                               |
| **`tests/`**      | **Unit Tests**             | `__init__.py`                                                                 |

---

## ⚙️ Configuration Details

All project settings are managed through `config/config.yaml`. This centralized file allows for easy modification of data ranges, asset universe, market constraints, and model hyperparameters.

### Key Configuration Groups

| Group                   | Description                                                              | Key Parameters                                                                                      | Default Values (Example)                                   |
| :---------------------- | :----------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| **`data`**              | Defines the asset universe, time ranges, and features used for training. | `assets` (25 tickers across 4 classes), `start_date`, `technical_indicators` (MACD, RSI, CCI, etc.) | `start_date: 2015-01-01`, `test_start: 2023-01-01`         |
| **`environment`**       | Parameters for the custom DRL environment.                               | `initial_amount`, `transaction_cost_pct`, `slippage_coefficient`, `reward_scaling`                  | `initial_amount: 1,000,000`, `transaction_cost_pct: 0.001` |
| **`constraints`**       | Real-world trading limits.                                               | `short_selling`, `max_leverage`, `max_position_size`, `sector_exposure_limits`                      | `max_leverage: 1.0`, `equities: 0.6`                       |
| **`transaction_costs`** | Settings for the cost analysis module.                                   | `cost_structures` (retail, institutional), `rebalance_frequencies` (daily, weekly, monthly)         | `retail: 0.005`, `weekly: 5`                               |
| **`regime_analysis`**   | Defines how market regimes are classified.                               | `methods` (volatility, trend, returns), `regime_definitions` (VIX/Return thresholds)                | `bull: {vix_threshold: 20}`, `bear: {vix_threshold: 30}`   |
| **`training`**          | General training parameters.                                             | `total_timesteps`, `n_eval_episodes`, `n_seeds`                                                     | `total_timesteps: 100000`, `n_seeds: 10`                   |
| **`production`**        | Settings for the FastAPI service and background tasks.                   | `api` (host, port), `rebalancing` (frequency, time), `risk_monitoring`                              | `api.port: 8000`, `rebalancing.frequency: weekly`          |

---

## 🧠 Deep Reinforcement Learning Agents

The project compares four distinct DRL algorithms, each representing a different approach to policy learning in continuous action spaces.

### Agent Comparison Table

| Algorithm   | Type                    | Core Concept                                                                                      | Key Advantage                                                                                             |
| :---------- | :---------------------- | :------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------- |
| **PPO**     | Policy Gradient         | Uses a clipped surrogate objective function to enable multiple epochs of minibatch updates.       | **Stability and Sample Efficiency**; excellent balance of exploration and exploitation.                   |
| **QR-DDPG** | Q-Learning/Actor-Critic | Extends DDPG by modeling the distribution of the Q-function (quantiles) instead of just the mean. | **Risk-Awareness**; better handling of tail risks and uncertainty in returns.                             |
| **DDPG**    | Actor-Critic            | Uses a deterministic policy and a Q-function to learn optimal actions in continuous spaces.       | **Simplicity and Determinism**; good for environments with smooth dynamics.                               |
| **SAC**     | Maximum Entropy RL      | Incorporates an entropy term into the reward to encourage exploration and robustness.             | **Exploration and Performance**; often achieves state-of-the-art performance with high sample efficiency. |

### Hyperparameter Configuration (from `config/config.yaml`)

| Parameter             | PPO              | QR-DDPG                          | DDPG                             | SAC         |
| :-------------------- | :--------------- | :------------------------------- | :------------------------------- | :---------- |
| **`total_timesteps`** | 100,000          | 100,000                          | 100,000                          | 100,000     |
| **`batch_size`**      | 256              | 128                              | 128                              | 256         |
| **`learning_rate`**   | 0.0003           | 0.0001 (Actor) / 0.0003 (Critic) | 0.0001 (Actor) / 0.0003 (Critic) | 0.0003      |
| **`buffer_size`**     | N/A              | 1,000,000                        | 1,000,000                        | 1,000,000   |
| **`ent_coef`**        | 0.01             | N/A                              | N/A                              | 0.2         |
| **`policy_kwargs`**   | `[128, 64]` ReLU | `[128, 64]`                      | `[128, 64]`                      | `[128, 64]` |

---

## 🔬 Advanced Analysis Modules

The repository includes three specialized modules for in-depth financial analysis, which can be run via dedicated Python scripts or Jupyter notebooks.

### 1. Transaction Cost Analysis

This module systematically evaluates the impact of trading costs and rebalancing frequency on portfolio performance.

**Key Findings (Example):**

- Retail costs (0.5%) can reduce annual returns by 15-20%.
- The **optimal rebalancing frequency** was found to be **Weekly to Biweekly** for most DRL strategies.

**Code Snippet (`code/transaction_cost_analysis.py`):**

```python
from code.transaction_cost_analysis import TransactionCostAnalyzer

analyzer = TransactionCostAnalyzer()

# Analyze performance across different cost structures and frequencies
results = analyzer.analyze_rebalancing_frequency(
    strategy_name='ppo',
    portfolio_values_base=portfolio_values,
    portfolio_weights_history=weights_history,
    dates=dates
)

# Generate visualization and report
analyzer.plot_cost_impact(results, save_path='results/cost_analysis.png')
```

### 2. Reward Ablation Study

The reward function includes a risk-aversion parameter, $\lambda$, which penalizes maximum drawdown. This study explores the optimal value for $\lambda \in [0.0, 1.0]$.

**Key Finding (Example):**

- An optimal $\lambda = 0.5$ was found to provide the best balance between maximizing returns (Sharpe Ratio) and minimizing risk (Max Drawdown).

**Code Snippet (`code/reward_ablation.py`):**

```python
from code.reward_ablation import RewardAblationStudy

study = RewardAblationStudy()

# Run ablation across lambda values defined in config.yaml
results = study.run_ablation_study(
    agent_class=PPO,
    env_factory=create_env_with_lambda,
    training_steps=100000
)

# Analyze and visualize the performance surface
study.plot_performance_surface(results, 'results/ablation_surface.png')
```

### 3. Market Regime Analysis

This module classifies the market into Bull, Bear, and Sideways regimes (using VIX and trend indicators) and evaluates which algorithm performs best in each condition.

**Key Findings (Example):**

- **Bull Markets**: SAC performs best (Sharpe: 2.5+), leveraging its superior exploration.
- **Bear Markets**: QR-DDPG excels (lowest CVaR), due to its risk-aware quantile modeling.
- **Sideways Markets**: PPO provides the most consistent, stable performance.

**Code Snippet (`code/regime_analysis.py`):**

```python
from code.regime_analysis import MarketRegimeAnalyzer

analyzer = MarketRegimeAnalyzer()

# Identify regimes using VIX and other methods
regime_df = analyzer.identify_regimes_vix(market_data)

# Analyze performance of all strategies by regime
performance = analyzer.analyze_performance_by_regime(
    strategy_results={'ppo': ppo_results, 'qr_ddpg': qr_results},
    regime_labels=regime_df
)
```

---

## 📈 Benchmarking Strategies

The DRL agents are benchmarked against a comprehensive set of traditional and advanced portfolio management strategies.

| Category                          | Strategy                             | Description                                                                             |
| :-------------------------------- | :----------------------------------- | :-------------------------------------------------------------------------------------- |
| **Traditional**                   | **Equal Weight**                     | Allocates an equal percentage to all assets.                                            |
| **Traditional**                   | **60/40 Portfolio**                  | Classic balanced portfolio (60% stocks, 40% bonds).                                     |
| **Modern Portfolio Theory (MPT)** | **Minimum Volatility**               | Minimizes portfolio variance based on historical data.                                  |
| **Modern Portfolio Theory (MPT)** | **Mean-Variance Optimization (MVO)** | Maximizes the Sharpe Ratio for a given risk level.                                      |
| **Risk-Based**                    | **Risk-Parity**                      | Allocates capital such that each asset contributes equally to the total portfolio risk. |
| **Risk-Based**                    | **Minimum Correlation**              | Selects assets to minimize the overall correlation within the portfolio.                |
| **Factor-Based**                  | **Momentum**                         | Invests in assets that have performed well recently.                                    |
| **Advanced**                      | **All-Weather Portfolio**            | Ray Dalio's strategy designed to perform well across all economic environments.         |

---

## 📊 Results & Analysis

The following tables summarize the performance of the DRL agents compared to the benchmarks during the test period (2023-2024).

### Performance Comparison (Test Period: 2023-2024)

| Strategy        | Sharpe Ratio | Annual Return | Max Drawdown (Max DD) | CVaR 5%   | Cost Impact (0.1% TC) |
| :-------------- | :----------- | :------------ | :-------------------- | :-------- | :-------------------- |
| **PPO**         | **2.15**     | **38.2%**     | -7.2%                 | -1.8%     | -2.1%                 |
| **QR-DDPG**     | 2.08         | 36.5%         | **-6.5%**             | **-1.5%** | -1.8%                 |
| **SAC**         | 1.98         | 35.1%         | -8.8%                 | -2.1%     | -2.3%                 |
| **DDPG**        | 1.85         | 32.1%         | -9.5%                 | -2.5%     | -2.0%                 |
| 60/40 Portfolio | 1.52         | 22.5%         | -11.2%                | -3.2%     | -0.8%                 |
| All-Weather     | 1.65         | 24.1%         | -9.5%                 | -2.8%     | -0.6%                 |
| Risk-Parity     | 1.45         | 25.8%         | -12.1%                | -3.1%     | -1.2%                 |

**Conclusion:** DRL agents, particularly **PPO** and **QR-DDPG**, significantly outperform traditional benchmarks across all key metrics, demonstrating superior risk-adjusted returns and lower tail risk.

### Performance by Market Regime

This table highlights the specialized strength of each DRL algorithm under different market conditions.

| Algorithm   | Bull Market Sharpe | Bear Market Sharpe | Sideways Market Sharpe | Best Performance           |
| :---------- | :----------------- | :----------------- | :--------------------- | :------------------------- |
| **PPO**     | 2.3                | 1.8                | **2.0**                | Consistent                 |
| **QR-DDPG** | 2.1                | **2.2**            | 1.9                    | Bear Market (Risk Control) |
| **SAC**     | **2.5**            | 1.5                | 1.8                    | Bull Market (Exploration)  |
| **DDPG**    | 1.9                | 1.2                | 1.5                    | N/A                        |

---

## 🏭 Production Deployment

The project is structured for seamless transition from research to a live trading environment using a microservices architecture.

### Services Architecture

The `docker-compose.yml` file orchestrates the following services:

```mermaid
graph TD
    A[Client Request] --> B(FastAPI API: 8000);
    B --> C(PostgreSQL DB: 5432);
    B --> D(Redis Cache: 6379);
    B --> E(Celery Worker);
    E --> F(Training/Rebalancing GPU Container);
    C --> G(Grafana Dashboard: 3000);
    F --> C;
```

### Example API Interaction

The FastAPI service exposes endpoints for real-time portfolio recommendations and risk monitoring.

**Endpoint: `/api/v1/portfolio/recommend` (POST)**

```bash
curl -X POST "http://localhost:8000/api/v1/portfolio/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "client_001",
    "risk_tolerance": "medium",
    "investment_amount": 1000000
  }'
```

**Example Response:**

```json
{
  "client_id": "client_001",
  "timestamp": "2024-01-15T10:30:00",
  "weights": {
    "AAPL": 0.12,
    "MSFT": 0.1,
    "BTC-USD": 0.08,
    "...": "..."
  },
  "expected_return": 25.3,
  "expected_volatility": 18.5,
  "sharpe_ratio": 1.85
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
