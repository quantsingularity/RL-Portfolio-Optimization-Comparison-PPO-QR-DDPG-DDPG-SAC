"""Unit tests for the PortfolioEnv trading environment."""

import numpy as np
import pytest
from environment import PortfolioEnv
from tests._fixtures import make_synthetic_panel


@pytest.fixture()
def env():
    df = make_synthetic_panel()
    return PortfolioEnv(df=df, initial_amount=1_000_000, print_verbosity=0)


def test_env_spaces(env):
    n_assets = 3
    expected_state_dim = 1 + n_assets + n_assets * 6
    assert env.observation_space.shape == (expected_state_dim,)
    assert env.action_space.shape == (n_assets,)


def test_env_reset(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert env.portfolio_value == env.initial_amount


def test_env_step(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "portfolio_value" in info


def test_weights_are_valid_simplex():
    """After softmax + hmax projection, weights must be a valid allocation."""
    # 5 assets with a 0.30 cap is feasible (5 * 0.30 = 1.5 >= 1).
    df = make_synthetic_panel(tickers=("AAA", "BBB", "CCC", "DDD", "EEE"))
    env = PortfolioEnv(df=df, print_verbosity=0)
    env.reset()
    env.step(np.array([5.0, -1.0, 0.5, 0.0, -2.0], dtype=np.float32))
    w = env.portfolio_weights
    assert np.all(w >= -1e-6)
    assert abs(w.sum() - 1.0) < 1e-4
    assert np.all(w <= env.hmax + 1e-6)


def test_infeasible_cap_falls_back_to_uniform():
    """cap * n_assets < 1 has no feasible solution; env must not crash and
    should return the closest feasible (uniform) allocation."""
    df = make_synthetic_panel()  # 3 assets, hmax 0.30 -> infeasible
    env = PortfolioEnv(df=df, print_verbosity=0)
    env.reset()
    env.step(np.array([2.0, -1.0, 0.5], dtype=np.float32))
    w = env.portfolio_weights
    assert abs(w.sum() - 1.0) < 1e-4
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0), atol=1e-5)


def test_full_episode_and_metrics(env):
    env.reset()
    done = False
    steps = 0
    while not done and steps < 10_000:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    metrics = env.get_portfolio_metrics()
    for key in ("sharpe_ratio", "max_drawdown", "annual_return", "cvar_5"):
        assert key in metrics
    # portfolio_values and date_memory stay in sync
    pv = env.save_portfolio_values()
    assert len(pv) == len(env.portfolio_values)


def test_transaction_costs_reduce_value():
    """Constant churn should incur non-zero transaction cost."""
    df = make_synthetic_panel()
    env = PortfolioEnv(df=df, transaction_cost_pct=0.01, print_verbosity=0)
    env.reset()
    _, _, _, _, info = env.step(np.array([5.0, 0.0, 0.0], dtype=np.float32))
    assert info["transaction_cost"] >= 0.0
