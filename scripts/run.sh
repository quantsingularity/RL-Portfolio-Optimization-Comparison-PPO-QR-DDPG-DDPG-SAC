#!/usr/bin/env bash
# =============================================================================
# Run script for DRL Portfolio Optimization (PPO / DDPG / SAC / QR-DDPG)
#
# Usage:
#   bash scripts/run.sh <command> [options]
#
# Commands:
#   setup       Install Python dependencies
#   test        Run the unit test suite
#   data        Run the data pipeline only (download + indicators + split)
#   train       Train all agents (PPO, DDPG, SAC, QR-DDPG) per config.yaml
#   evaluate    Evaluate trained agents vs classical benchmarks
#   api         Launch the FastAPI production service
#   demo        Quick offline smoke run on synthetic data (~2 min, no network)
#   all         setup -> test -> train -> evaluate
#
# Notes:
#   - Live market data requires network access to Yahoo Finance. If
#     unavailable, the pipeline automatically falls back to synthetic data
#     (also selectable via data.use_synthetic_data: true in config.yaml).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$ROOT_DIR/code"

export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python3}"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

cmd="${1:-}"
shift || true

case "$cmd" in
    setup)
        echo "[run.sh] Installing dependencies..."
        "$PYTHON" -m pip install -r "$CODE_DIR/requirements.txt"
        ;;

    test)
        echo "[run.sh] Running unit tests..."
        cd "$CODE_DIR"
        "$PYTHON" -m pytest tests/ -v "$@"
        ;;

    data)
        echo "[run.sh] Running data pipeline..."
        cd "$CODE_DIR"
        "$PYTHON" data/data_processor.py "$@"
        ;;

    train)
        echo "[run.sh] Training all DRL agents..."
        cd "$CODE_DIR"
        "$PYTHON" training/train.py "$@"
        ;;

    evaluate)
        echo "[run.sh] Evaluating agents vs benchmarks..."
        cd "$CODE_DIR"
        "$PYTHON" evaluation/evaluate.py "$@"
        ;;

    api)
        echo "[run.sh] Launching FastAPI production service..."
        cd "$CODE_DIR"
        "$PYTHON" production/api.py "$@"
        ;;

    demo)
        echo "[run.sh] Demo: synthetic-data pipeline + QR-DDPG smoke run..."
        cd "$CODE_DIR"
        "$PYTHON" - <<'PY'
import warnings
warnings.filterwarnings("ignore")

import yaml
import numpy as np
from data import DataProcessor
from environment import PortfolioEnv
from agents import QRDDPGAgent

cfg = yaml.safe_load(open("config/config.yaml"))
cfg["data"]["use_synthetic_data"] = True
cfg["data"]["start_date"] = "2022-01-01"
cfg["data"]["end_date"] = "2022-06-30"
cfg["data"]["train_start"] = "2022-01-01"
cfg["data"]["train_end"] = "2022-04-30"
cfg["data"]["test_start"] = "2022-05-01"
cfg["data"]["test_end"] = "2022-06-30"

processor = DataProcessor(cfg)
train, test = processor.process_all()
print(f"\nSynthetic universe: {train['tic'].nunique()} assets")

env = PortfolioEnv(df=train, initial_amount=1e6, print_verbosity=0)
sd = env.observation_space.shape[0]
ad = env.action_space.shape[0]
agent = QRDDPGAgent(state_dim=sd, action_dim=ad, n_quantiles=20, device="cpu")

obs, _ = env.reset()
for i in range(300):
    action = agent.select_action(obs, noise=0.1)
    nobs, reward, term, trunc, _ = env.step(action)
    agent.replay_buffer.push(obs, action, reward, nobs, float(term or trunc))
    obs = nobs
    if len(agent.replay_buffer) > 64:
        agent.update(64)
    if term or trunc:
        obs, _ = env.reset()

tenv = PortfolioEnv(df=test, initial_amount=1e6, print_verbosity=0)
obs, _ = tenv.reset()
done = False
while not done:
    action = agent.select_action(obs, noise=0.0)
    obs, _, term, trunc, _ = tenv.step(action)
    done = term or trunc

metrics = tenv.get_portfolio_metrics()
print("\nQR-DDPG out-of-sample metrics (synthetic data):")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
print("\nDemo finished successfully.")
PY
        ;;

    all)
        bash "$0" setup
        bash "$0" test
        bash "$0" train
        bash "$0" evaluate
        ;;

    -h|--help|help|"")
        usage
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        usage
        exit 1
        ;;
esac
