"""Unit tests for the DDPG and QR-DDPG agents."""

import numpy as np
import torch
from agents import DDPGAgent, QRDDPGAgent
from agents.agents import QuantileCritic, ReplayBuffer

STATE_DIM = 12
ACTION_DIM = 3


def _fill_buffer(agent, n: int = 200) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n):
        agent.replay_buffer.push(
            rng.normal(size=STATE_DIM).astype(np.float32),
            rng.uniform(-1, 1, size=ACTION_DIM).astype(np.float32),
            float(rng.normal()),
            rng.normal(size=STATE_DIM).astype(np.float32),
            0.0,
        )


def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(150):
        buf.push(
            np.ones(STATE_DIM, dtype=np.float32) * i,
            np.zeros(ACTION_DIM, dtype=np.float32),
            0.0,
            np.ones(STATE_DIM, dtype=np.float32),
            False,
        )
    assert len(buf) == 100  # capacity respected
    states, actions, rewards, next_states, dones = buf.sample(16)
    assert states.shape == (16, STATE_DIM)
    assert actions.shape == (16, ACTION_DIM)
    assert rewards.shape == (16,)


def test_ddpg_action_bounds():
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, device="cpu")
    action = agent.select_action(np.zeros(STATE_DIM, dtype=np.float32), noise=0.3)
    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_ddpg_update_changes_weights():
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, device="cpu")
    _fill_buffer(agent)
    before = [p.detach().clone() for p in agent.critic.parameters()]
    agent.update(batch_size=32)
    after = list(agent.critic.parameters())
    changed = any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert changed, "critic parameters should change after an update"


def test_qr_ddpg_quantile_critic_shape():
    critic = QuantileCritic(STATE_DIM, ACTION_DIM, n_quantiles=25)
    s = torch.randn(8, STATE_DIM)
    a = torch.randn(8, ACTION_DIM)
    out = critic(s, a)
    assert out.shape == (8, 25)


def test_qr_ddpg_update_runs():
    agent = QRDDPGAgent(STATE_DIM, ACTION_DIM, n_quantiles=20, device="cpu")
    _fill_buffer(agent)
    agent.update(batch_size=32)  # should not raise
    action = agent.select_action(np.zeros(STATE_DIM, dtype=np.float32), noise=0.0)
    assert action.shape == (ACTION_DIM,)


def test_quantile_huber_loss_ordering():
    """
    The pairwise quantile Huber loss is not zero even for identical
    predicted and target quantile sets (it compares all quantile pairs),
    but it must be strictly smaller for a perfect prediction than for a
    shifted one.
    """
    tau = torch.FloatTensor([(i + 0.5) / 10 for i in range(10)])
    q = torch.sort(torch.randn(4, 10), dim=1).values
    loss_perfect = QRDDPGAgent._quantile_huber_loss(q, q.clone(), tau)
    loss_shifted = QRDDPGAgent._quantile_huber_loss(q, q + 5.0, tau)
    assert loss_perfect.item() < loss_shifted.item()
