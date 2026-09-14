"""PettingZoo ``ParallelEnv`` adapter over :class:`hamlet.core.HamletCore`.

Agents are named ``agent_0`` to ``agent_{N-1}`` and always act together.
Episodes end by truncation at ``cfg.episode_ticks``; ``terminations`` are
always ``False``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from hamlet.config import N_ACTIONS, HamletConfig
from hamlet.core import HamletCore


def agent_ids(n: int) -> list[str]:
    """Agent names ``agent_0 .. agent_{n-1}``, in core index order."""
    return [f"agent_{i}" for i in range(n)]


class HamletParallelEnv(ParallelEnv):
    """Parallel multi-agent view of one :class:`HamletCore`.

    ``reset(seed, options)`` accepts ``options={"init_range": (lo, hi)}`` to
    override the initial state distribution. ``render()`` returns ``None``.
    """

    metadata = {"render_modes": [], "name": "hamlet_v0", "is_parallelizable": True}

    def __init__(self, cfg: Optional[HamletConfig] = None, seed: Optional[int] = 0, render_mode: Optional[str] = None) -> None:
        self.cfg = cfg if cfg is not None else HamletConfig()
        self.core = HamletCore(self.cfg, seed)
        self.render_mode = render_mode
        self.possible_agents: list[str] = agent_ids(self.cfg.N)
        self.agents: list[str] = []
        self._obs_space = Box(0.0, 1.0, shape=(self.cfg.obs_dim,), dtype=np.float32)
        self._act_space = Discrete(N_ACTIONS)

    def observation_space(self, agent: str) -> Box:
        return self._obs_space

    def action_space(self, agent: str) -> Discrete:
        return self._act_space

    def _split(self, arr: np.ndarray) -> dict[str, Any]:
        return {a: arr[i] for i, a in enumerate(self.possible_agents)}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        init_range = options.get("init_range") if options else None
        obs = self.core.reset(seed=seed, init_range=init_range)
        self.agents = list(self.possible_agents)
        return self._split(obs), {a: {} for a in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}
        arr = np.array([actions[a] for a in self.possible_agents], dtype=np.int64)
        obs, rew, info = self.core.step(arr)
        truncated = bool(info["truncated"])
        agents = self.agents
        observations = self._split(obs)
        rewards = {a: float(rew[i]) for i, a in enumerate(agents)}
        terminations = {a: False for a in agents}
        truncations = {a: truncated for a in agents}
        infos = {a: {"t": info["t"], "day": info["day"], "t_day": info["t_day"]} for a in agents}
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
