"""Throughput of the simulation core under a random policy.

Runs two full episodes for each population size and prints ticks per second
and agent-steps per second (ticks times agents), including the cost of
building observations and drawing random actions.
"""
from __future__ import annotations

import time

import numpy as np

from hamlet.config import HamletConfig
from hamlet.core import HamletCore
from hamlet.policies import RandomPolicy

POPULATIONS = (4, 8, 16)
EPISODES = 2


def run(n_agents: int, episodes: int) -> tuple[float, float]:
    """Return ``(ticks_per_second, agent_steps_per_second)`` for ``n_agents``."""
    cfg = HamletConfig(n_agents=n_agents)
    core = HamletCore(cfg, seed=0)
    policy = RandomPolicy(np.random.default_rng(0))
    ticks = 0
    start = time.perf_counter()
    for episode in range(episodes):
        obs = core.reset(episode)
        for _ in range(cfg.episode_ticks):
            actions, _ = policy.act(obs, core)
            obs, _, _ = core.step(actions)
            ticks += 1
    elapsed = time.perf_counter() - start
    return ticks / elapsed, ticks * n_agents / elapsed


def main() -> None:
    print(f"{'N':>4} {'ticks/s':>10} {'agent-steps/s':>15}")
    for n in POPULATIONS:
        tps, aps = run(n, EPISODES)
        print(f"{n:>4} {tps:>10.0f} {aps:>15.0f}")


if __name__ == "__main__":
    main()
