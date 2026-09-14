"""PettingZoo parallel API conformance."""
from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test

from hamlet.config import N_ACTIONS, HamletConfig
from hamlet.pz_env import HamletParallelEnv


def test_parallel_api():
    env = HamletParallelEnv(HamletConfig(), seed=0)
    parallel_api_test(env, num_cycles=1500)


def test_spaces_and_truncation():
    cfg = HamletConfig(n_agents=3, n_days=2)
    env = HamletParallelEnv(cfg, seed=0)
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents) == {"agent_0", "agent_1", "agent_2"}
    space = env.observation_space("agent_0")
    assert space.shape == (cfg.obs_dim,) and space.contains(obs["agent_0"])
    assert env.action_space("agent_0").n == N_ACTIONS
    for t in range(cfg.episode_ticks):
        obs, rew, term, trunc, info = env.step({a: 0 for a in env.agents})
        assert not any(term.values())
        assert all(trunc.values()) == (t == cfg.episode_ticks - 1)
        assert all(space.contains(o) for o in obs.values())
    assert env.agents == []
    assert env.render() is None
    obs, _ = env.reset(seed=0, options={"init_range": (0.2, 0.3)})
    assert np.all(env.core.E >= 0.2) and np.all(env.core.E <= 0.3)
