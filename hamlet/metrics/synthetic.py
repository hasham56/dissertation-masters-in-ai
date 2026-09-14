"""Synthetic behaviour logs with known structure, shared by the tests and the report scripts.

``make_log`` builds a LOG_COLUMNS frame from arrays; ``schedule_only_process``
generates a population whose zone choice depends only on its own states and
on the schedule (opening hours and night), never on the hour inside a window,
optionally with a planted within-window dependence. It is the reference
process for validating the clock-information null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hamlet.config import (CANTEEN, FARM, HOME, LOG_COLUMNS, MARKET, SOCIAL, TRANSIT, N_ACTIONS,
                          HamletConfig)

# The synthetic condition label follows the configured population so the fixtures and the
# real logs agree on the name. It is a label only; nothing in this module reads N from it.
_CFG = HamletConfig()

T = 240
N = 8
DAYS = 4

def make_log(
    rng: np.random.Generator,
    n_agents: int = N,
    n_days: int = DAYS,
    ticks_per_day: int = T,
    zone: np.ndarray | None = None,
    active: np.ndarray | None = None,
    states: np.ndarray | None = None,
    queued: np.ndarray | None = None,
    informed: np.ndarray | None = None,
    logp: np.ndarray | None = None,
    reward: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per (tick, agent), tick-major. Optional arrays are (T_total, N) shaped."""
    total = n_days * ticks_per_day
    n = total * n_agents
    t = np.repeat(np.arange(total), n_agents)
    agent = np.tile(np.arange(n_agents), total)
    z = rng.integers(0, 6, size=n) if zone is None else np.asarray(zone).reshape(n)
    act = np.ones(n, dtype=bool) if active is None else np.asarray(active).reshape(n).astype(bool)
    st = rng.random((n, 3)) if states is None else np.asarray(states).reshape(n, 3)
    d = ((1 - st) ** 2).sum(axis=1)
    q = np.zeros(n, dtype=bool) if queued is None else np.asarray(queued).reshape(n).astype(bool)
    inf = np.zeros(n, dtype=bool) if informed is None else np.asarray(informed).reshape(n).astype(bool)
    lp = np.full((n, N_ACTIONS), np.log(1.0 / N_ACTIONS)) if logp is None else np.asarray(logp).reshape(n, N_ACTIONS)
    rew = -d if reward is None else np.asarray(reward).reshape(n)
    frame = pd.DataFrame({
        "seed": 10_000, "condition": f"A-S1-N{_CFG.n_agents}", "checkpoint": "none", "episode": 0,
        "t": t, "day": t // ticks_per_day, "t_day": t % ticks_per_day, "agent": agent,
        # decision: synthetic logs have no journeys, so every tick is a decision tick
        "x": 0, "y": 0, "zone": z, "action": z, "decision": True, "active": act, "queued": q,
        "E": st[:, 0], "F": st[:, 1], "C": st[:, 2],
        # World v2 columns. The synthetic log is a schema fixture, not a simulation: the tanks are
        # split so that fast + slow reproduces F exactly, and no meal or company is asserted.
        "fast_tank": st[:, 1] * 0.5, "slow_tank": st[:, 1] * 0.5,
        "meal_type": "none", "company_others": 0,
        "W": 20.0, "M": 0.7, "D": d, "rest_idle_cost": 0.0,
        "coins_spent_food": 0.0, "coins_spent_drink": 0.0, "drank": False,
        "reward": rew, "effort_penalty": 0.0, "informed": inf,
        **{f"logp{k}": lp[:, k] for k in range(N_ACTIONS)},
    })
    return frame[LOG_COLUMNS]


SCHED_DAYS = 5
SCHED_LEAK = 0.01
SCHED_GAIN = 4.0
SCHED_STICK = 0.9
SCHED_LOW = 0.5
SCHED_P = 0.8
SCHED_P_MARKET = 0.5
SCHED_P_SERVE = 0.8
SCHED_FALLBACK = (HOME, CANTEEN, SOCIAL, TRANSIT)
SCHED_SEEDS = (32, 33, 35)
PLANT_BIN = (120, 150)     # 12:00-15:00, the fourth hour bin, inside the work window
PLANT_PROB = 0.8


def schedule_only_process(
    rng: np.random.Generator,
    plant: int | None = None,
    leak: float | None = None,
    stick: float | None = None,
    n_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(T_total, N) zones and (T_total, N, 3) states of the schedule-only process.

    With ``plant`` set, every agent-day is forced to that zone for the whole
    of ``PLANT_BIN`` with probability ``PLANT_PROB``: a within-window
    dependence that the schedule strata do not explain.
    """
    leak = SCHED_LEAK if leak is None else leak
    stick = SCHED_STICK if stick is None else stick
    n_days = SCHED_DAYS if n_days is None else n_days
    total = n_days * T
    zone = np.empty((total, N), dtype=np.int64)
    states = np.empty((total, N, 3))
    s = rng.random((N, 3))
    z = rng.integers(0, 6, size=N)
    serve = np.array([HOME, CANTEEN, SOCIAL])
    forced = np.zeros(N, dtype=bool)
    for t in range(total):
        td = t % T
        work, market, night = 80 <= td < 180, 60 <= td < 200, td >= 220 or td < 60
        if plant is not None and td == PLANT_BIN[0]:
            forced = rng.random(N) < PLANT_PROB
        for i in range(N):
            if rng.random() > stick:
                k = int(np.argmin(s[i]))
                if s[i, k] < SCHED_LOW:
                    z[i] = serve[k] if rng.random() < SCHED_P_SERVE else rng.integers(0, 6)
                elif night and rng.random() < SCHED_P:
                    z[i] = HOME
                elif work and rng.random() < SCHED_P:
                    z[i] = FARM
                elif market and rng.random() < SCHED_P_MARKET:
                    z[i] = MARKET
                else:
                    z[i] = rng.choice(SCHED_FALLBACK)
            zz = plant if (plant is not None and forced[i] and PLANT_BIN[0] <= td < PLANT_BIN[1]) else z[i]
            zone[t, i] = zz
            occupied = np.array([zz == HOME, zz == CANTEEN, zz == SOCIAL], dtype=float)
            s[i] = np.minimum(1.0, (1 - leak) * s[i] + leak * SCHED_GAIN * occupied)
            states[t, i] = s[i]
    return zone, states


def schedule_only_log(
    seed: int, plant: int | None = None, leak: float | None = None, stick: float | None = None, n_days: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = SCHED_DAYS if n_days is None else n_days
    zone, states = schedule_only_process(rng, plant, leak=leak, stick=stick, n_days=n_days)
    return make_log(rng, n_days=n_days, zone=zone, states=states)

