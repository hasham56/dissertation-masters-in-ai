"""Trajectory regression against the tagged world.

The fixture stores a hash of every agent's position and needs at every tick
over two simulated days, per baseline policy, for the world as it stood at
the ``world-v2`` pin. Prior pins: ``study-n9`` (N = 9 with three Food Street
seats), ``travel-v1`` (N = 8, fixed-cost travel) and ``jobset-v1`` (before
fixed-cost travel).

World v2 changes almost every hash by construction. The bar
replaced the all-or-nothing social rule with a staffed room where a solo visit
earns half; satiety became the clipped sum of a fast and a slow tank, each with
its own drain; both meals now cost coins and are refused when the agent cannot
pay; the jobs, the market, Food Street and the bar all gained or changed opening
hours, with eviction at close for the last three and an unpaid lunch hour at the
jobs; a sleep bonus follows an evening with company; a restlessness charge falls
on an agent that sits at home once rested, outside the night; and the observation
grew by two entries for the tanks. Re-pinned again after a fix that
corrected the tank mapping (a market meal fills the slow tank, a Food Street meal
the fast one, which had been implemented the wrong way round) and calibrated
``fast_tank_drain`` 0.005 to 0.0015, ``fast_tank_fill`` 0.024 to 0.035 and ``wage``
1.0 to 4.0. Re-pinned again for v2-lite: opening hours
moved to the jobs only (07:00-19:00), the two-tank economy went behind
``two_tank_economy`` defaulting off, and with it off satiety returns to v1's
single level in a free canteen. The bar with its staff, floor and slope, the
sleep bonus and the restlessness charge all stay. Re-pinned again
for economy v3: food and drink are bought rather than given, so the
canteen charges (``free_canteen`` off, Food Street 8.0, the market stall 3.0) and
a drink costs 3.0, with no drink meaning no company; the market and Food Street
have their own hours again; the farm feeds whoever works it up to
``farm_feed_cap`` and pays ``farm_wage_share`` of the office wage while draining
``farm_energy_mult`` times the ordinary work energy; a need is served only once
it falls below ``GREEDY_SERVE_TRIGGER`` lowered by laziness, where before the
ceiling was ``greedy_base_threshold``; and the schedulers gained a money rule
that sends an agent below a day's budget to a job, to the farm when hungry and
to the office otherwise. Any change to the dynamics, the observation builder,
the RNG draw order or a baseline policy changes a hash. A change that is
meant to be behaviour-preserving (the neutral population of a trait system,
a refactor) must leave every hash intact; a deliberate world change
regenerates the fixture with::

    uv run python -m tests.test_regression_world --update

and records the reason in the fixture's ``reason`` field.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from hamlet.config import HamletConfig
from hamlet.core import HamletCore
from hamlet.policies import GreedyClockPolicy, GreedyClockWorkPolicy, GreedyStatePolicy, RandomPolicy

FIXTURE = Path(__file__).parent / "fixtures" / "world_trajectories.json"
SEEDS = (0, 10000)
DAYS = 2


def policies() -> dict[str, callable]:
    return {
        "RANDOM": lambda: RandomPolicy(np.random.default_rng(123)),
        "GREEDY-STATE": GreedyStatePolicy,
        "GREEDY-CLOCK": GreedyClockPolicy,
        "GREEDY-CLOCK-WORK": GreedyClockWorkPolicy,
    }


def trajectory_hash(policy, seed: int) -> str:
    cfg = HamletConfig()
    core = HamletCore(cfg, seed)
    obs = core.reset(seed)
    h = hashlib.sha256()
    for _ in range(DAYS * cfg.ticks_per_day):
        a, _ = policy.act(obs, core)
        obs, rew, _ = core.step(a)
        h.update(core.pos.astype(np.int64).tobytes())
        h.update(np.round(np.stack([core.E, core.F, core.C, core.W, core.M]), 9).tobytes())
        h.update(a.astype(np.int64).tobytes())
        h.update(np.round(rew, 9).astype(np.float64).tobytes())
    return h.hexdigest()


def current() -> dict[str, str]:
    return {f"{name}/seed{seed}": trajectory_hash(make(), seed) for name, make in policies().items() for seed in SEEDS}


def test_world_trajectories_unchanged():
    expected = json.loads(FIXTURE.read_text())["hashes"]
    got = current()
    assert got == expected, {k: (expected.get(k), got.get(k)) for k in got if expected.get(k) != got.get(k)}


if __name__ == "__main__":
    if "--update" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps({
            "tag": "study-n9", "days": DAYS, "seeds": list(SEEDS),
            "reason": [
                "The population moved from 8 agents to 9 and the capacities followed their "
                "registered formulas: canteen ceil(N/4) 2 -> 3, each job ceil(N/2) 4 -> 5. "
                "Every trajectory changes by construction, because "
                "the observation is 46 + N wide and one more agent draws from the world "
                "stream. The previous pins were travel-v1 and jobset-v1.",
                "GREEDY-CLOCK-WORK only, re-pinned separately: where aptitude does not separate "
                "the two jobs it now breaks the tie with a per-episode random ranking instead of "
                "by distance from the agent's home, a vestige of the world that fixed-cost travel "
                "removed. Homes sit in a column in agent order, so that term handed "
                "identical agents a fixed identity-linked division of labour and the H3 positive "
                "control's neutral population read a cross-episode ARI of 0.84 against its 0.2 "
                "ceiling; it now reads -0.08 and the control passes. Only the two "
                "GREEDY-CLOCK-WORK hashes move: the other six are byte-identical, because the "
                "ranking is drawn from the core's bit-generator state without consuming it and no "
                "world draw shifts.",
            ],
            "hashes": current()}, indent=2) + "\n")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(current(), indent=2))
