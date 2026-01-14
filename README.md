# Reinforcement Learning for Risk-Aware Portfolio Optimization

A comprehensive implementation of Deep Reinforcement Learning (DRL) algorithms for dynamic portfolio optimization with explicit risk management. This project implements and compares four state-of-the-art DRL algorithms: **PPO**, **QR-DDPG**, **DDPG**, and **SAC** against traditional benchmark strategies.

## 📋 Overview

This project implements the research paper "_Reinforcement Learning for Risk-Aware Portfolio Optimization: A Comparative Study of PPO, QR-DDPG, DDPG, and SAC under Market Uncertainty_".

### Key Features

- **Risk-Aware MDP Formulation**: Custom reward function with explicit maximum drawdown penalty
- **Distributional RL**: Implementation of Quantile Regression DDPG (QR-DDPG) for tail-risk optimization
- **Comprehensive Benchmarks**: Comparison against MVO, Risk-Parity, Minimum Volatility, Momentum, and Equal-Weight strategies
- **Statistical Validation**: ANOVA and Tukey's HSD tests for significance testing
- **Interpretability**: SHAP analysis for policy explanation
- **Multi-Asset Universe**: 25 assets across equities, cryptocurrencies, commodities, and fixed income

## 🎯 Research Highlights

- **Best Performance**: PPO achieves Sharpe Ratio of 2.15 ± 0.05
- **Best Risk Management**: QR-DDPG achieves lowest CVaR of -1.5% ± 0.1%
- **Statistical Significance**: PPO and QR-DDPG significantly outperform traditional benchmarks (p < 0.01)
- **Adaptive Strategy**: Agents learn to respond to macro-level risk indicators (VIX) for strategic rebalancing

## 🏗️ Project Structure

```
rl_portfolio_project/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data_processor.py        # Data fetching and preprocessing
│   ├── environment.py           # Custom RL environment (MDP)
│   ├── agents.py                # DRL agents (DDPG, QR-DDPG)
│   ├── benchmarks.py            # Traditional benchmark strategies
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation and testing
│   └── figure_generation.py     # Visualization generation
├── data/                        # Downloaded market data (auto-created)
├── results/                     # Evaluation results and figures
│   ├── figures/                 # Generated visualizations
│   └── logs/                    # Training logs
├── models/                      # Trained model checkpoints
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/quantsingularity/RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC
cd RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Quick Start

#### 1. Data Preparation

```bash
cd src
python data_processor.py
```

This will download historical data for all 25 assets from Yahoo Finance (2015-2024).

#### 2. Train DRL Agents

```bash
python train.py
```

This trains all four DRL agents (PPO, DDPG, SAC, QR-DDPG) with multiple random seeds.

**Note**: Full training with 10 seeds takes ~6-8 hours on GPU. For quick testing, the script uses 2 seeds by default.

#### 3. Evaluate All Strategies

```bash
python evaluate.py
```

Evaluates trained DRL agents and backtests benchmark strategies on the test period (2023-2024).

#### 4. Generate Figures

```bash
python figure_generation.py
```

Generates all 7 figures from the research paper.

## 📊 Results

### Performance Comparison (Test Period: 2023-2024)

| Strategy             | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) | CVaR (5%) (%) |
| -------------------- | ----------------- | ------------ | ---------------- | ------------- |
| **PPO (Risk-Aware)** | 38.2 ± 1.1        | 2.15 ± 0.05  | -7.2 ± 0.3       | -1.8 ± 0.1    |
| **QR-DDPG**          | 36.5 ± 1.2        | 2.08 ± 0.06  | -6.5 ± 0.2       | -1.5 ± 0.1    |
| **SAC**              | 35.1 ± 1.3        | 1.98 ± 0.06  | -8.8 ± 0.5       | -2.1 ± 0.1    |
| **DDPG**             | 31.5 ± 1.5        | 1.78 ± 0.07  | -10.5 ± 0.8      | -2.5 ± 0.2    |
| Risk-Parity (RP)     | 25.8 ± 0.0        | 1.45 ± 0.00  | -12.1 ± 0.0      | -3.1 ± 0.0    |
| MVO                  | 22.1 ± 0.0        | 1.25 ± 0.00  | -15.2 ± 0.0      | -3.8 ± 0.0    |
| Min Volatility (MVP) | 18.5 ± 0.0        | 1.05 ± 0.00  | -8.5 ± 0.0       | -2.0 ± 0.0    |
| Equal-Weight (EW)    | 15.5 ± 0.0        | 0.85 ± 0.00  | -20.1 ± 0.0      | -5.0 ± 0.0    |

### Key Findings

1. **Superior Performance**: DRL agents significantly outperform traditional strategies in risk-adjusted returns
2. **Risk Management**: QR-DDPG's distributional approach achieves the best tail-risk management
3. **Adaptability**: Agents learn to respond proactively to market volatility (VIX spikes)
4. **Statistical Validation**: ANOVA confirms significant differences (F=15.8, p<0.001)

## 🧪 Methodology

### 1. Markov Decision Process (MDP) Formulation

**State Space** (S):

- Asset prices (25 assets)
- Technical indicators: MACD, RSI, CCI, Bollinger Bands
- Macroeconomic factors: VIX index
- Current portfolio weights
- Portfolio value

**Action Space** (A):

- Continuous portfolio weight changes: Δw ∈ [-1, 1]^25

**Reward Function** (R):

```
R_t = log(V_t / V_{t-1}) - λ * MaxDrawdown_t - TransactionCost_t
```

- λ = 0.5 (maximum drawdown penalty coefficient)
- Explicitly promotes risk-averse behavior

### 2. Algorithms Implemented

#### PPO (Proximal Policy Optimization)

- On-policy actor-critic method
- Clipped surrogate objective for stable updates
- Best overall risk-adjusted performance

#### QR-DDPG (Quantile Regression DDPG)

- Off-policy distributional RL
- Models full return distribution (50 quantiles)
- Optimizes CVaR for tail-risk management

#### DDPG (Deep Deterministic Policy Gradient)

- Off-policy actor-critic for continuous actions
- Deterministic policy with exploration noise

#### SAC (Soft Actor-Critic)

- Off-policy maximum entropy RL
- Temperature-controlled exploration
- Robust across different market conditions

### 3. Network Architecture

All agents use two-layer feed-forward networks:

- Hidden layers: [128, 64] neurons
- Activation: ReLU
- Output: Tanh (for bounded actions)

### 4. Training Hyperparameters

| Hyperparameter         | PPO  | QR-DDPG | DDPG | SAC  |
| ---------------------- | ---- | ------- | ---- | ---- |
| Learning Rate (Actor)  | 3e-4 | 1e-4    | 1e-4 | 3e-4 |
| Learning Rate (Critic) | 3e-4 | 3e-4    | 3e-4 | 3e-4 |
| Batch Size             | 256  | 128     | 128  | 256  |
| Gamma (γ)              | 0.99 | 0.99    | 0.99 | 0.99 |
| Buffer Size            | -    | 1M      | 1M   | 1M   |
| # Quantiles (N)        | -    | 50      | -    | -    |

## 📈 Figures

The project generates all figures from the research paper:

1. **Figure 1**: Cumulative Portfolio Returns (2023-2024)
2. **Figure 2**: PPO Sensitivity to Drawdown Penalty (λ)
3. **Figure 3**: SHAP Feature Importance Analysis
4. **Figure 4**: Dynamic Portfolio Weights Trajectory
5. **Figure 5**: Statistical Significance (Tukey's HSD)
6. **Figure 6**: Ablation Study Results
7. **Figure 7**: Distribution of Daily Returns (KDE)

Run `python figure_generation.py` to generate all figures.

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- **Asset Universe**: Add/remove tickers
- **Date Ranges**: Training and testing periods
- **Hyperparameters**: Learning rates, batch sizes, etc.
- **Risk Parameters**: Drawdown penalty (λ), transaction costs
- **Environment**: Initial capital, rebalancing frequency

## 📝 Usage Examples

### Train Individual Agents

```python
from src.train import TrainDRLAgents

trainer = TrainDRLAgents()
trainer.prepare_data()

# Train only PPO
ppo_model = trainer.train_ppo(seed=0)

# Train only QR-DDPG
qr_ddpg_agent = trainer.train_qr_ddpg(seed=0)
```

### Evaluate Specific Strategy

```python
from src.evaluate import EvaluateStrategies

evaluator = EvaluateStrategies()
evaluator.load_data()

# Evaluate DRL agents
drl_results = evaluator.evaluate_drl_agents()

# Evaluate benchmarks
benchmark_results = evaluator.evaluate_benchmarks()
```

### Custom Backtesting

```python
from src.benchmarks import BacktestBenchmark

backtester = BacktestBenchmark(df=test_data)

# Backtest custom strategy
results = backtester.backtest_strategy('risk_parity')
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

## 📚 Research Paper

This implementation is based on:

**Title**: Reinforcement Learning for Risk-Aware Portfolio Optimization: A Comparative Study of PPO, QR-DDPG, DDPG, and SAC under Market Uncertainty

**Author**: Abrar Ahmed

**Date**: December 09, 2025

**Abstract**: This paper presents a rigorous, risk-aware framework for dynamic portfolio optimization through comparative study of four advanced DRL algorithms with explicit tail-risk management using distributional RL methods.

## 🙏 Acknowledgments

- **FinRL Library**: AI4Finance Foundation
- **Stable-Baselines3**: DLR-RM
- **Research Papers**: Schulman et al. (PPO), Dabney et al. (QR-DQN), Lillicrap et al. (DDPG), Haarnoja et al. (SAC)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
