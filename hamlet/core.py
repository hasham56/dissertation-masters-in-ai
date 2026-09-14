"""The Hamlet simulation core: a NumPy village with homeostatic agents.

This module holds the whole environment dynamics and nothing else. It imports
only ``numpy``, :mod:`hamlet.config` and :mod:`hamlet.traits`; the PettingZoo
adapter wraps it, and every policy (hand-written or learned) drives
it through :meth:`HamletCore.step`. All agents are updated with array
operations; there is no per-agent Python loop inside ``step``.

Units
-----
* positions are integer grid tiles, origin top-left, ``x`` to the right and
  ``y`` downwards;
* time is counted in ticks (``ticks_per_day`` ticks make one day);
* internal states ``E``, ``F``, ``C`` and mood ``M`` are dimensionless levels
  in ``[0, 1]``; wealth ``W`` is in coins; drive ``D`` is a weighted sum of
  squared deficits and lies in ``[0, w_E + w_F + w_C]``;
* ``skill`` is a dimensionless level in ``[0, skill_max]`` per agent and job;
  ``effort_penalty`` is in reward units (the same unit as ``reward``).

Traits
------
Every agent carries six read-only traits (:mod:`hamlet.traits`), drawn at
construction from ``traits_seed`` and never from ``rng``:

* ``E -= metabolism * (energy_drain + energy_work_drain * [active at a job])``
* ``E += energy_rest_gain * (1 + night_rest_bonus * in_window) * [resting at home]``
  with ``in_window = is_night(t_day - chronotype * ticks_per_hour)`` per agent
  (chronotype in hours, wrapped modulo ``ticks_per_day``); the out-of-window
  rest rate is therefore ``1 / (1 + night_rest_bonus)`` of the in-window rate;
* ``F -= appetite * satiety_drain`` (canteen and market gains unchanged);
* ``coins = wage * productivity * aptitude[job] * (1 + skill[job])`` per active
  job tick, then ``skill[job] += learning_rate * (1 - skill[job] / skill_max)``
  and ``skill[other jobs] *= 1 - forgetting_rate``;
* ``reward -= effort_coef * laziness * effort`` where ``effort`` is 1 on a tick
  the agent worked or changed tile and ``effort_coef`` is
  ``effort_penalty_coef`` (arms A, B, E) or ``effort_penalty_coef * wage``
  (arms C, D). The penalty is logged as ``effort_penalty``. With laziness
  above zero the difference reward form no longer telescopes to a function of
  the end states.

The neutral population (every trait at its neutral value) reproduces the
trait-free dynamics bit for bit.

Observation features
--------------------
The layout is fixed by ``config.OBS``. Every feature is scaled into
``[0, 1]`` so that one flat ``Box(0, 1)`` describes the observation space.
The clock block therefore stores ``(1 + sin) / 2`` and ``(1 + cos) / 2`` of
the day phase rather than the raw sine and cosine, which would take values
in ``[-1, 1]``. The transform is affine and loses nothing.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hamlet.config import (
    ACTION_NAMES,
    CANTEEN,
    FARM,
    GO_CANTEEN,
    GO_FARM,
    GO_HOME,
    GO_MARKET,
    GO_OFFICE,
    GO_SOCIAL,
    HOME,
    HOME_X_STEP,
    HOME_Y_FIRST,
    HOME_Y_STEP,
    IDLE,
    JOBS,
    LOG_COLUMNS,
    MARKET,
    N_ACTIONS,
    OBS,
    OFFICE,
    PUBLIC_ZONES,
    SOCIAL,
    TRANSIT,
    ZONE_RECTS,
    HamletConfig,
)
from hamlet.traits import TraitArrays, sample_traits

# Index of the three activities tracked by ``ticks_since``.
REST, MEAL, SOCIALISE = 0, 1, 2

# Zone requested by each action; IDLE requests nothing.
_ACTION_TO_ZONE = np.array(
    [HOME, FARM, OFFICE, CANTEEN, SOCIAL, MARKET, -1], dtype=np.int64
)
assert _ACTION_TO_ZONE[GO_HOME] == HOME and _ACTION_TO_ZONE[GO_FARM] == FARM
assert _ACTION_TO_ZONE[GO_OFFICE] == OFFICE and _ACTION_TO_ZONE[GO_CANTEEN] == CANTEEN
assert _ACTION_TO_ZONE[GO_SOCIAL] == SOCIAL and _ACTION_TO_ZONE[GO_MARKET] == MARKET
assert _ACTION_TO_ZONE[IDLE] == -1
assert len(ACTION_NAMES) == N_ACTIONS == len(_ACTION_TO_ZONE)
# World v2 meal kinds, in the order of HamletCore.meal_type (0 none, 1 fast, 2 slow).
# Where the meal came from, in the order of HamletCore.meal_type.
_MEAL_NAMES = ("none", "food_street", "market", "farm")

# Zones resolved in step 2, in id order; TRANSIT is never a target.
_REQUESTABLE_ZONES = (HOME, FARM, OFFICE, CANTEEN, SOCIAL, MARKET)


class HamletCore:
    """Vectorised simulation of ``cfg.n_agents`` agents on an open grid.

    Parameters
    ----------
    cfg:
        The world constants. ``cfg.validate()`` is called once here.
    seed:
        Seed of ``self.rng``, the only source of world randomness. Construction
        performs ``reset(seed)``, so a core is ready to step immediately and
        ``HamletCore(cfg, s)`` equals ``HamletCore(cfg, s).reset(s)``.
    traits_seed:
        Run seed of the trait draw (``hamlet.traits.sample_traits``); defaults
        to ``seed`` (and to 0 when ``seed`` is ``None``). Trait draws never
        touch ``self.rng``.
    traits:
        Explicit :class:`hamlet.traits.TraitArrays` for the ``N`` agents,
        overriding the draw (identity permutations, tests).

    Public attributes, length ``N`` unless stated: ``t, day, t_day, pos (N, 2),
    home (N, 2), E, F, C, W, M, D, zone, active, queued, informed,
    prev_action, ticks_since (N, 3), coins_earned, purchased, effort_penalty,
    skill (N, n_jobs), traits (TraitArrays), traits_seed, weights (3,), rng``.
    """

    def __init__(
        self,
        cfg: HamletConfig,
        seed: Optional[int] = None,
        traits_seed: Optional[int] = None,
        traits: Optional[TraitArrays] = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.rng: np.random.Generator = np.random.default_rng(seed)
        n = cfg.N
        g = cfg.grid

        self.traits_seed: int = int(traits_seed if traits_seed is not None else (seed if seed is not None else 0))
        if traits is None:
            traits = TraitArrays.from_traits(
                sample_traits(cfg.population, n, self.traits_seed, normalise=cfg.normalise_aptitude)
            )
        self.set_traits(traits)
        self._job_index = np.full(N_ACTIONS, -1, dtype=np.int64)
        for k, job in enumerate(JOBS):
            self._job_index[_ACTION_TO_ZONE == job] = k

        # Rectangle bounds per zone id, inclusive; HOME and TRANSIT rows are
        # placeholders (HOME targets are per agent, TRANSIT is never a target).
        lo = np.zeros((TRANSIT + 1, 2), dtype=np.int64)
        hi = np.zeros((TRANSIT + 1, 2), dtype=np.int64)
        for zone, (x0, y0, x1, y1) in ZONE_RECTS.items():
            lo[zone] = (x0, y0)
            hi[zone] = (x1, y1)
        self._rect_lo = lo
        self._rect_hi = hi
        self._public = np.array(PUBLIC_ZONES, dtype=np.int64)
        self._public_lo = lo[self._public]  # (5, 2)
        self._public_hi = hi[self._public]  # (5, 2)

        # Zone id of every tile (HOME is resolved per agent at lookup time).
        zone_grid = np.full((g, g), TRANSIT, dtype=np.int64)
        for zone, (x0, y0, x1, y1) in ZONE_RECTS.items():
            zone_grid[x0 : x1 + 1, y0 : y1 + 1] = zone
        self._zone_grid = zone_grid

        # Capacity used for the occupancy feature: N when unlimited.
        self._occ_denominator = np.array(
            [cfg.capacity(z) if cfg.capacity(z) is not None else n for z in PUBLIC_ZONES],
            dtype=np.float64,
        )

        # Effective drive weights; social weight is zero when social is off.
        self.weights = np.array(
            [cfg.w_energy, cfg.w_satiety, cfg.w_social if cfg.social_on else 0.0],
            dtype=np.float64,
        )
        self._agent_id_block = (
            np.eye(n, dtype=np.float32) if cfg.symmetry == "S1" else np.zeros((n, n), np.float32)
        )
        self._home_tiles = self._home_layout(n, g, cfg.home_x)
        self._dist_scale = 2.0 * (g - 1)
        self._pos_scale = float(g - 1)
        self._arange = np.arange(n)

        self.reset(seed)

    def set_traits(self, traits: TraitArrays) -> None:
        """Install ``traits`` (validated, one row per agent) and rebuild the observation block."""
        n = self.cfg.N
        if traits.n != n or traits.aptitude.shape != (n, len(JOBS)):
            raise ValueError(f"traits must describe {n} agents and {len(JOBS)} jobs")
        self.traits = traits.validate()
        vectors = traits.to_vectors()
        self._trait_block = vectors if self.cfg.obs_include_traits else np.zeros_like(vectors)

    @staticmethod
    def _home_layout(n: int, grid: int, home_x: int) -> np.ndarray:
        """Home tiles ``(n, 2)`` in agent-index order before permutation.

        Column ``home_x`` holds rows ``HOME_Y_FIRST + HOME_Y_STEP * k``; when
        ``n`` exceeds what one column holds (``HamletConfig.validate`` allows
        up to twice as many), the overflow continues ``HOME_X_STEP`` tiles to
        the right with the same rows.
        """
        rows = (grid - 1 - HOME_Y_FIRST) // HOME_Y_STEP + 1
        k = np.arange(n, dtype=np.int64)
        x = home_x + HOME_X_STEP * (k // rows)
        y = HOME_Y_FIRST + HOME_Y_STEP * (k % rows)
        return np.stack([x, y], axis=1)

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: Optional[int] = None,
        init_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Start a new episode and return the first observation, ``(N, obs_dim)``.

        ``seed`` re-seeds ``self.rng``; ``None`` keeps drawing from the current
        stream so that consecutive episodes differ. ``init_range`` overrides
        ``(init_state_low, init_state_high)`` for the initial ``E, F, C`` draw.
        Traits and ``skill`` are per run, not per episode: traits are kept and
        ``skill`` returns to zero.
        """
        cfg = self.cfg
        n = cfg.N
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        lo, hi = init_range if init_range is not None else (cfg.init_state_low, cfg.init_state_high)

        self.t: int = 0
        # HOME is the agent's own zone (capacity one), so there is no home tile to draw; the anchors
        # below are for the map only.
        self.home = self._home_tiles[: n].copy()
        self.pos = self.home.copy()
        states = self.rng.uniform(lo, hi, size=(3, n))
        self.E = states[0].astype(np.float64)
        self.C = states[2].astype(np.float64)
        # World v2: satiety is the clipped sum of two tanks rather than a level of its own. The
        # tanks are drawn independently so that F starts in the same range as v1's single draw;
        # states[1] is consumed but unused, so the rest of the random stream is unchanged.
        #
        # init_range must reach the tanks. The init-dispersion evaluation passes (init_narrow,
        # init_wide) work by widening or narrowing this draw, and H3d reads the result; tanks that
        # ignored it would leave satiety identically distributed in every pass. Each tank takes
        # half the requested range, so fast + slow spans exactly (lo, hi) as the single v1 level did.
        if init_range is not None:
            f_lo = s_lo = lo / 2.0
            f_hi = s_hi = hi / 2.0
        else:
            f_lo, f_hi = cfg.init_fast_low, cfg.init_fast_high
            s_lo, s_hi = cfg.init_slow_low, cfg.init_slow_high
        if cfg.two_tank_economy:
            self.fast = np.minimum(self.rng.uniform(f_lo, f_hi, size=n), cfg.tank_cap)
            self.slow = np.minimum(self.rng.uniform(s_lo, s_hi, size=n), cfg.tank_cap)
        else:
            self.fast = np.zeros(n, dtype=np.float64)
            self.slow = states[1].astype(np.float64)      # v1's single satiety draw
        self.F = np.minimum(1.0, self.fast + self.slow)
        self.W = np.full(n, cfg.init_wealth, dtype=np.float64)
        self.M = np.full(n, cfg.init_mood, dtype=np.float64)
        self.D = self._drive()
        self.zone = np.full(n, HOME, dtype=np.int64)      # every agent starts in its own HOME zone
        self._transit_left = np.zeros(n, dtype=np.int64)  # 0 = at a zone and free to decide
        self._transit_target = np.full(n, HOME, dtype=np.int64)
        self._held_action = np.full(n, IDLE, dtype=np.int64)
        self.decision = np.ones(n, dtype=bool)
        self.active = np.zeros(n, dtype=bool)
        self.queued = np.zeros(n, dtype=bool)
        self.informed = np.zeros(n, dtype=bool)
        if cfg.gossip:
            self.informed[self.rng.integers(n)] = True
        self.prev_action = np.full(n, -1, dtype=np.int64)
        self.ticks_since = np.zeros((n, 3), dtype=np.int64)
        self.coins_earned = np.zeros(n, dtype=np.float64)
        self.purchased = np.zeros(n, dtype=bool)
        # World v2 bookkeeping, all reported in the log.
        # meal_type: 0 none, 1 fast (a market meal), 2 slow (a Food Street tick).
        self.meal_type = np.zeros(n, dtype=np.int64)
        self.company_others = np.zeros(n, dtype=np.int64)
        # The sleep-bonus flag expires at this tick; -1 means no flag is live. A separate array
        # remembers that the flag was live when the agent arrived home, because the bonus lasts
        # for the whole stay even after the flag itself has expired.
        self._sleep_bonus_until = np.full(n, -1, dtype=np.int64)
        self._sleep_bonus_stay = np.zeros(n, dtype=bool)
        # Whether the agent was resting at home on the previous tick, so that arrival can be
        # told from a stay in progress.
        self._was_home = np.zeros(n, dtype=bool)
        # Consecutive ticks spent at HOME while rested and outside the world's night, and the
        # restlessness charge those ticks have earned. The charge is a reward term, not state:
        # it never enters D.
        # Whether the agent held a Food Street seat on the previous tick, so a new sitting can be
        # told from one already paid for.
        self._seated = np.zeros(n, dtype=bool)
        self._at_market = np.zeros(n, dtype=bool)
        self._drinking = np.zeros(n, dtype=bool)
        self.farm_fed = np.zeros(n, dtype=np.float64)
        self.coins_spent_food = np.zeros(n, dtype=np.float64)
        self.coins_spent_drink = np.zeros(n, dtype=np.float64)
        self.drank = np.zeros(n, dtype=bool)
        self._rest_idle_ticks = np.zeros(n, dtype=np.int64)
        self.rest_idle_cost = np.zeros(n, dtype=np.float64)
        self.effort_penalty = np.zeros(n, dtype=np.float64)
        self.skill = np.zeros((n, len(JOBS)), dtype=np.float64)
        self._active_zone = np.full(n, -1, dtype=np.int64)
        self._last_reward = np.zeros(n, dtype=np.float32)
        # Tick, day and tick-of-day at which the last applied actions were taken.
        self._step_t = 0
        self._step_day = 0
        self._step_t_day = 0
        return self._observe()

    # ------------------------------------------------------------- properties
    @property
    def day(self) -> int:
        """Day index, ``t // ticks_per_day``."""
        return self.t // self.cfg.ticks_per_day

    @property
    def t_day(self) -> int:
        """Tick of day, ``t % ticks_per_day``."""
        return self.t % self.cfg.ticks_per_day

    @property
    def can_decide(self) -> np.ndarray:
        """``bool (N,)``: whether the next action passed to :meth:`step` will be used.

        False while a journey is running, where the held action applies and the passed one is
        ignored. Read this before stepping; ``self.decision`` records the same flag for the tick
        that has just been applied, and is what the log column carries.
        """
        return self._transit_left == 0

    @property
    def truncated(self) -> bool:
        """True once ``t == episode_ticks``; the episode must then be reset."""
        return self.t >= self.cfg.episode_ticks

    def night_window(self, t_day: Optional[int] = None) -> np.ndarray:
        """Per-agent night-rest window, ``bool (N,)``: ``is_night`` of the chronotype-shifted clock.

        Agent ``i`` is in its window when ``cfg.is_night((t_day - chronotype_i *
        ticks_per_hour) mod ticks_per_day)``; a positive chronotype (hours)
        moves the window later. With every chronotype at zero this equals
        ``cfg.is_night(t_day)`` for everyone.
        """
        cfg = self.cfg
        t_day = self.t_day if t_day is None else int(t_day)
        shifted = np.mod(t_day - self.traits.chronotype * cfg.ticks_per_hour, cfg.ticks_per_day)
        a, b = cfg.night
        return (shifted >= a) | (shifted < b)

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance the world by one tick.

        Parameters
        ----------
        actions:
            Integer array ``(N,)`` of action ids in ``[0, N_ACTIONS)``.

        Returns
        -------
        obs:
            ``float32 (N, obs_dim)`` observation for the next decision.
        rew:
            ``float32 (N,)`` reward of the transition.
        info:
            ``{"t", "day", "t_day", "truncated"}`` after the tick.
        """
        cfg = self.cfg
        n = cfg.N
        if self.t >= cfg.episode_ticks:
            raise RuntimeError("episode is over (t == episode_ticks); call reset()")
        actions = np.array(actions, dtype=np.int64).reshape(n)  # copy: kept as prev_action
        if actions.min() < 0 or actions.max() >= N_ACTIONS:
            raise ValueError("action ids must lie in [0, N_ACTIONS)")

        t_day = self.t_day
        day = self.day
        in_window = self.night_window(t_day)
        traits = self.traits
        jobs_shock = cfg.shock == "jobs_closed_d4" and day == cfg.shock_day
        jobs_open = cfg.job_is_open(t_day) and not jobs_shock
        # Open is not the same as paying: no wage is earned over the unpaid lunch hour.
        jobs_pay = cfg.job_pays(t_day) and not jobs_shock
        energy_mult = (
            cfg.shock_energy_mult if (cfg.shock == "energy_x2_d4" and day >= cfg.shock_day) else 1.0
        )
        D_old = self.D

        # 1. travel: a journey between two different zones costs cfg.travel_ticks ticks in TRANSIT,
        #    then one arrival tick at the destination. The policy is consulted only when no journey
        #    is running (self.decision); during a journey the action is held and the passed action
        #    is ignored. self._transit_left counts the ticks still owed: > 1 is a TRANSIT tick, == 1
        #    is the arrival tick, == 0 is a decision tick.
        decision = self._transit_left == 0
        self.decision = decision
        effective = np.where(decision, actions, self._held_action)
        self.prev_action = effective
        target_zone = _ACTION_TO_ZONE[effective]

        starting = decision & (effective != IDLE) & (target_zone != self.zone)
        if starting.any():
            self._held_action = np.where(starting, effective, self._held_action)
            self._transit_target = np.where(starting, target_zone, self._transit_target)
            # travel_ticks TRANSIT ticks (this one included) plus the arrival tick
            self._transit_left = np.where(starting, cfg.travel_ticks + 1, self._transit_left)

        travelling = self._transit_left > 1
        arriving = self._transit_left == 1
        self.zone = np.where(travelling, TRANSIT, np.where(arriving, self._transit_target, self.zone))
        moved = travelling.copy()          # the effort term: a journey costs effort, standing still does not
        self._transit_left = np.maximum(self._transit_left - 1, 0)

        # 2. requests, capacity and opening hours -> active / queued.
        #    A traveller requests nothing; everyone else requests the zone it now stands in.
        requesting = (~travelling) & (effective != IDLE) & (target_zone == self.zone)
        # World v2 opening hours. A zone that is shut admits nobody, and anyone who was active or
        # queued there when it shut is simply not active this tick: that is the eviction. Because an
        # evicted agent is standing still rather than travelling, its _transit_left is already 0, so
        # it takes a decision tick immediately and chooses where to go next.
        #
        # The jobs are the exception the spec calls out. They shut at 19:00 like anything else, but
        # they do not evict over the unpaid lunch hour: an agent at its desk at 12:00 keeps its seat
        # and its place in the queue through to 13:00 and simply earns nothing, which is what
        # `jobs_pay` below controls. Eviction and payment are separate questions here.
        zone_open = {
            HOME: True,
            FARM: jobs_open,
            OFFICE: jobs_open,
            CANTEEN: cfg.canteen_is_open(t_day),
            SOCIAL: cfg.bar_is_open(t_day),
            MARKET: cfg.market_is_open(t_day),
        }
        active = np.zeros(n, dtype=bool)
        queued = np.zeros(n, dtype=bool)
        order = self.rng.permutation(n)
        for zone in _REQUESTABLE_ZONES:
            req = requesting & (target_zone == zone)
            if not zone_open[zone] or not req.any():
                continue
            cap = cfg.capacity(zone)
            if zone == HOME or cap is None:
                # HOME is the agent's own tile, so its capacity of one is implicit.
                active |= req
                continue
            holders = req & (self._active_zone == zone)
            n_free = max(int(cap) - int(holders.sum()), 0)
            newcomers = order[(req & ~holders)[order]]
            active[holders] = True
            active[newcomers[:n_free]] = True
            queued[newcomers[n_free:]] = True
        self.active = active
        self.queued = queued
        self._active_zone = np.where(active, target_zone, -1)

        act_home = active & (target_zone == HOME)
        # Active at any job: the two jobs differ only in place and capacity.
        act_job = np.zeros(n, dtype=bool)
        for job in JOBS:
            act_job |= active & (target_zone == job)
        # Index into JOBS of the job worked this tick (0 where no job; masked by act_job below).
        job_k = np.maximum(np.where(act_job, self._job_index[actions], 0), 0)
        act_farm = active & (target_zone == FARM)
        act_canteen = active & (target_zone == CANTEEN)
        act_social = active & (target_zone == SOCIAL)
        act_market = active & (target_zone == MARKET)

        # 3. meals. Both kinds cost coins and both are refused outright when the agent cannot pay:
        #    no gain, no debit, and meal_type "none". A refusal is silent, exactly as if the agent
        #    had sat there doing nothing, which is what makes the wage and the prices worth
        #    calibrating against each other.
        #
        #    A staple bought at the market fills the slow half outright, for market_price.
        #    Food Street charges food_street_price once, on the tick the seat is taken; the sitting
        #    then fills the fast half to its cap over the following ticks at no further cost.
        #    Leaving and re-sitting is a new sitting and a new charge, so dithering is paid for.
        if cfg.two_tank_economy:
            buy = act_market & (self.W >= cfg.market_price)
            new_sitting = act_canteen & ~self._seated & (self.W >= cfg.food_street_price)
            seated_on = act_canteen & (self._seated | new_sitting)
            self.refused_meal = (act_market & ~buy) | (act_canteen & ~seated_on)
            self.W = self.W - cfg.market_price * buy - cfg.food_street_price * new_sitting
            # The v3 sources do not exist in the two-tank world; meal_type still needs them.
            at_market_on = buy
            farm_feeding = np.zeros(n, dtype=bool)
        else:
            # Economy v3. Three ways to eat, each charged once per sitting and each refused
            # outright when the purse is short:
            #   the market stall, market_price, fills at eating speed while the agent stays;
            #   Food Street, food_street_price, fills satiety outright on the tick the seat is taken;
            #   the farm, free, fills at eating speed while the agent works it in job hours.
            # free_canteen restores v2-lite, where the CANTEEN zone cost nothing.
            new_market = act_market & ~self._at_market & (self.W >= cfg.market_price)
            at_market_on = act_market & (self._at_market | new_market)
            if cfg.free_canteen:
                new_sitting = act_canteen & ~self._seated
                seated_on = act_canteen
            else:
                new_sitting = act_canteen & ~self._seated & (self.W >= cfg.food_street_price)
                seated_on = act_canteen & (self._seated | new_sitting)
            # The farm feeds only up to farm_feed_cap; past it the agent keeps working and keeps
            # earning, but stops being fed and must buy the rest of its satiety elsewhere.
            farm_feeding = act_farm & cfg.farm_feeds & jobs_pay & (self.slow < cfg.farm_feed_cap)
            self.refused_meal = (act_market & ~at_market_on) | (act_canteen & ~seated_on)
            buy = new_sitting                      # a Food Street seat is the instant meal
            self.coins_spent_food = (cfg.market_price * new_market
                                     + (0.0 if cfg.free_canteen else cfg.food_street_price) * new_sitting)
            self.W = self.W - self.coins_spent_food
            self._at_market = at_market_on
        self._seated = seated_on
        # A staple fills the slow half outright; a sitting fills the fast half a little each tick
        # until it reaches the cap. Both truncate at cfg.tank_cap and neither is refunded, so
        # turning up already full is simply wasted money.
        if cfg.two_tank_economy:
            self.slow = np.where(buy, cfg.tank_cap, self.slow)
            self.fast = np.minimum(cfg.tank_cap, self.fast + cfg.fast_tank_fill * seated_on)
        else:
            # v2-lite keeps one satiety level, carried in the slow tank so the log schema is
            # unchanged; the fast tank stays at zero and F is the slow tank alone.
            # A productive farm tick feeds the worker while it earns.
            # farm_feeding already carries the cap; the gain must respect it too.
            farm_fed = farm_feeding * cfg.farm_feed_rate
            self.farm_fed = farm_fed
            # Food Street fills satiety outright; the stall and the farm fill at eating speed.
            filled = np.minimum(1.0, self.slow + cfg.canteen_gain_v1 * at_market_on + farm_fed)
            self.slow = np.where(seated_on & ~cfg.free_canteen, 1.0,
                                 np.minimum(1.0, filled + cfg.canteen_gain_v1 * seated_on * cfg.free_canteen))
        self.purchased = buy
        # meal_type names the tank the meal fills, not where it was bought: a market meal is the
        # "slow" one and a Food Street meal the "fast" one.
        # meal_type names where the meal came from: 1 Food Street, 2 the market stall, 3 the farm.
        self.meal_type = np.where(seated_on, 1, np.where(at_market_on, 2,
                                  np.where(farm_feeding, 3, 0))).astype(np.int64)

        # 4. state deltas, productivity, wealth, mood, elapsed-time counters.
        E_pre, F_pre = self.E, self.F
        productivity = np.minimum(1.0, E_pre / cfg.productivity_floor) * np.minimum(
            1.0, F_pre / cfg.productivity_floor
        )
        if cfg.mood_coupling:
            productivity = productivity * (0.5 + 0.5 * self.M)

        # World v2, the bar. `others` counts the other villagers active at the bar this tick. The
        # staff are a fixed presence with no needs, no decisions and no row in the log, so they are
        # not villagers and never count here; their whole effect is that bar_solo_share is 0.5
        # rather than nothing, so an agent drinking alone is not drinking in an empty room.
        # Economy v3: a drink is charged once, when the agent becomes active at the bar, and is
        # refused outright if the purse is short. A refused agent buys nothing and gains nothing.
        new_drink = act_social & ~self._drinking & (self.W >= cfg.drink_price)
        drinking = act_social & (self._drinking | new_drink)
        self.coins_spent_drink = cfg.drink_price * new_drink
        self.W = self.W - self.coins_spent_drink
        self.drank = drinking
        self._drinking = drinking
        act_social = drinking                    # no drink, no company

        others = np.where(act_social, max(int(act_social.sum()) - 1, 0), 0).astype(np.int64)
        self.company_others = others
        company = np.zeros(n, dtype=np.float64)
        if cfg.social_on:
            mult = cfg.bar_solo_share + cfg.bar_company_step * np.minimum(others, cfg.bar_company_cap)
            company = cfg.social_gain * mult * act_social

        # World v2, the sleep bonus. Any company gain at all sets a flag that expires
        # sleep_bonus_ticks later. The bonus applies if the flag was live at the moment the agent
        # arrived home, and then lasts for the whole stay even once the flag has expired: it is one
        # good night's sleep bought by the evening, not a rate that ticks down mid-sleep.
        earned_company = company > 0.0
        self._sleep_bonus_until = np.where(earned_company, self.t + cfg.sleep_bonus_ticks,
                                           self._sleep_bonus_until)
        arrived_home = act_home & ~self._was_home
        flag_live = self._sleep_bonus_until > self.t
        self._sleep_bonus_stay = np.where(arrived_home, flag_live, self._sleep_bonus_stay & act_home)
        rest_mult = np.where(self._sleep_bonus_stay & act_home, cfg.sleep_bonus_mult, 1.0)

        # Working the farm costs more energy than the office when farm_feeds is on.
        work_mult = np.where(act_farm & cfg.farm_feeds, cfg.farm_energy_mult, 1.0)
        E = E_pre - cfg.energy_drain * energy_mult * traits.metabolism
        E = E - cfg.energy_work_drain * energy_mult * traits.metabolism * work_mult * act_job
        E = E + cfg.energy_rest_gain * (1.0 + cfg.night_rest_bonus * in_window) * rest_mult * act_home
        # The two hunger tanks drain independently; satiety is their clipped sum. Appetite scales
        # both, so a hungry trait is hungry in both tanks.
        if cfg.two_tank_economy:
            self.fast = np.clip(self.fast - cfg.fast_tank_drain * traits.appetite, 0.0, cfg.tank_cap)
            self.slow = np.clip(self.slow - cfg.slow_tank_drain * traits.appetite, 0.0, cfg.tank_cap)
        else:
            self.fast = np.zeros(n, dtype=np.float64)
            self.slow = np.clip(self.slow - cfg.satiety_drain_v1 * traits.appetite, 0.0, 1.0)
        C = self.C - cfg.social_drain + company
        self.E = np.minimum(np.maximum(E, 0.0), 1.0)
        self.F = np.minimum(1.0, self.fast + self.slow)
        self.C = np.minimum(np.maximum(C, 0.0), 1.0)
        self._was_home = act_home.copy()

        aptitude = traits.aptitude[self._arange, job_k]
        skill = self.skill[self._arange, job_k]
        # No wage over the unpaid lunch hour, though the agent keeps its seat.
        # The farm and the office pay differently when farm_feeds is on: job_k is 0 for FARM and
        # 1 for OFFICE, following JOBS.
        wage_of_job = np.where(job_k == 0, cfg.farm_wage, cfg.office_wage)
        self.coins_earned = wage_of_job * productivity * act_job * aptitude * (1.0 + skill) * jobs_pay
        self.W = self.W + self.coins_earned
        # Skill grows at the job worked this tick and, if forgetting_rate > 0, decays elsewhere.
        if cfg.forgetting_rate > 0.0:
            decayed = self.skill * (1.0 - cfg.forgetting_rate)
            decayed[self._arange[act_job], job_k[act_job]] = skill[act_job]
            self.skill = decayed
        grown = skill + traits.learning_rate * (1.0 - skill / cfg.skill_max)
        self.skill[self._arange, job_k] = np.where(act_job, np.minimum(grown, cfg.skill_max), skill)
        # World v2: restlessness at home. An agent that has restored its energy and stays put is
        # idling, so each further tick costs a little. Three things reset the counter: leaving
        # HOME, energy falling below rest_idle_energy, and 22:00. The world's night is used here,
        # not the agent's chronotype-shifted window, because the rule is about the hours the
        # village keeps rather than the agent's own body clock.
        # Two exemptions: never at night, and never while nothing the agent could use is open.
        # Charging an agent for sitting at home when there is nowhere to go would punish it for the
        # village's hours rather than for idling.
        nothing_open = not (cfg.job_is_open(t_day) or cfg.market_is_open(t_day)
                            or cfg.canteen_is_open(t_day) or cfg.bar_is_open(t_day))
        if cfg.is_night(t_day) or nothing_open:
            self._rest_idle_ticks[:] = 0
            self.rest_idle_cost = np.zeros(n, dtype=np.float64)
        else:
            settled = act_home & (self.E >= cfg.rest_idle_energy)
            self._rest_idle_ticks = np.where(settled, self._rest_idle_ticks + 1, 0)
            beyond = np.maximum(self._rest_idle_ticks - cfg.rest_idle_after, 0)
            self.rest_idle_cost = np.minimum(cfg.rest_idle_cap, cfg.rest_idle_cost * beyond)

        self.D = self._drive()
        self.M = cfg.mood_tau * self.M + (1.0 - cfg.mood_tau) * (1.0 - self.D / 3.0)

        self.ticks_since = self.ticks_since + 1
        self.ticks_since[act_home, REST] = 0
        self.ticks_since[seated_on | buy, MEAL] = 0
        self.ticks_since[act_social, SOCIALISE] = 0

        # 5. reward, minus the effort penalty (working or changing tile, scaled by laziness).
        if cfg.homeostatic_reward:
            if cfg.reward_form == "level":
                rew = -self.D
            else:
                rew = cfg.difference_scale * (D_old - self.D)
            effort_coef = cfg.effort_penalty_coef
        else:
            rew = cfg.income_scale * self.coins_earned
            effort_coef = cfg.effort_penalty_coef * cfg.wage
        effort = act_job | moved
        self.effort_penalty = effort_coef * traits.laziness * effort
        # The reward carries the restlessness charge; D does not. Dashboards report D, so they
        # show the drive the agent actually carries, and the reward stream stays decomposable
        # into -(D), the effort penalty and this.
        rew = (rew - self.effort_penalty - self.rest_idle_cost).astype(np.float32)
        self._last_reward = rew

        # 6. gossip: the bit spreads among everyone active at SOCIAL together.
        if cfg.gossip and int(act_social.sum()) >= 2 and self.informed[act_social].any():
            self.informed = self.informed | act_social

        # 7. shocks were applied above (jobs_shock in step 2, energy_mult in step 4).
        # 8. advance the clock and observe.
        self._step_t, self._step_day, self._step_t_day = self.t, day, t_day
        self.t += 1
        obs = self._observe()
        info = {
            "t": self.t,
            "day": self.day,
            "t_day": self.t_day,
            "truncated": self.t == cfg.episode_ticks,
        }
        return obs, rew, info

    # ---------------------------------------------------------------- helpers
    def _drive(self) -> np.ndarray:
        """Drive ``D = w_E(1-E)^2 + w_F(1-F)^2 + w_C(1-C)^2`` (dimensionless)."""
        w = self.weights
        return w[0] * (1.0 - self.E) ** 2 + w[1] * (1.0 - self.F) ** 2 + w[2] * (1.0 - self.C) ** 2

    def _zone_of(self, pos: np.ndarray) -> np.ndarray:
        """Zone id per agent; HOME only on the agent's own home tile."""
        zone = self._zone_grid[pos[:, 0], pos[:, 1]]
        at_home = (pos == self.home).all(axis=1)
        return np.where(at_home, HOME, zone)

    def _target_tiles(self, actions: np.ndarray, target_zone: np.ndarray) -> np.ndarray:
        """Nearest tile of each agent's requested zone, ``(N, 2)``.

        Public zones: the agent's position clamped into the rectangle.
        GO_HOME: the agent's own home tile. IDLE: the current position.
        """
        safe_zone = np.where(target_zone < 0, TRANSIT, target_zone)
        target = np.minimum(np.maximum(self.pos, self._rect_lo[safe_zone]), self._rect_hi[safe_zone])
        go_home = (actions == GO_HOME)[:, None]
        idle = (actions == IDLE)[:, None]
        target = np.where(go_home, self.home, target)
        target = np.where(idle, self.pos, target)
        return target

    def _public_distances(self) -> np.ndarray:
        """Manhattan distance in tiles from each agent to the five public zones, ``(N, 5)``."""
        nearest = np.minimum(np.maximum(self.pos[:, None, :], self._public_lo[None]), self._public_hi[None])
        return np.abs(nearest - self.pos[:, None, :]).sum(axis=2)

    def _open_flags(self, t_day: int) -> np.ndarray:
        """Open flag per public zone in ``PUBLIC_ZONES`` order; a shock is not announced here."""
        cfg = self.cfg
        flags = np.ones(len(PUBLIC_ZONES), dtype=np.float32)
        for k, zone in enumerate(PUBLIC_ZONES):
            if zone in JOBS:
                flags[k] = float(cfg.job_is_open(t_day))
            elif zone == MARKET:
                flags[k] = float(cfg.market_is_open(t_day))
        return flags

    def _observe(self) -> np.ndarray:
        """Build the ``(N, obs_dim)`` float32 observation following ``config.OBS``."""
        cfg = self.cfg
        n = cfg.N
        obs = np.zeros((n, cfg.obs_dim), dtype=np.float32)
        t_day = self.t_day

        if cfg.arm == "E":
            obs[:, OBS["states"]] = np.minimum(self.ticks_since / cfg.proxy_scale, 1.0)
        elif cfg.states_visible:
            obs[:, OBS["states"]] = np.stack([self.E, self.F, self.C], axis=1)
        if not cfg.social_on:
            obs[:, OBS["states"].stop - 1] = 0.0

        # World v2: the two hunger tanks are shown separately as well as through F, because the
        # agent cannot otherwise tell a full fast tank about to empty from a full slow one. They
        # follow the same visibility rule as the states: hidden in arm B, replaced by the elapsed
        # proxies in arm E.
        if cfg.arm == "E":
            obs[:, OBS["tanks"]] = np.minimum(
                self.ticks_since[:, [MEAL, MEAL]] / cfg.proxy_scale, 1.0)
        elif cfg.states_visible:
            obs[:, OBS["tanks"]] = np.stack([self.fast, self.slow], axis=1)

        obs[:, OBS["wealth"]] = np.minimum(self.W / cfg.wealth_obs_scale, 1.0)
        if cfg.clock_visible:
            phase = 2.0 * np.pi * t_day / cfg.ticks_per_day
            obs[:, OBS["clock"]] = (0.5 * (1.0 + np.sin(phase)), 0.5 * (1.0 + np.cos(phase)))
        # xy, dist and home_dist stay zero: position and Manhattan distance left the simulation
        # when travel became a fixed cost. The entries keep their slots so the layout is
        # unchanged at 48 + N (config.OBS), zero-filled exactly as a removed input is elsewhere.
        counts = np.bincount(self._active_zone[self.active], minlength=TRANSIT + 1)
        obs[:, OBS["occ"]] = counts[self._public] / self._occ_denominator
        obs[:, OBS["open"]] = self._open_flags(t_day)

        zone_block = OBS["zone"]
        obs[self._arange, zone_block.start + self.zone] = 1.0
        acted = self.prev_action >= 0
        prev_block = OBS["prev_action"]
        obs[self._arange[acted], prev_block.start + self.prev_action[acted]] = 1.0
        obs[:, OBS["active"]] = self.active
        obs[:, OBS["traits"]] = self._trait_block
        obs[:, OBS["agent_id"]] = self._agent_id_block

        assert obs.shape == (n, cfg.obs_dim)
        return obs

    # ---------------------------------------------------------------- logging
    def log_columns(
        self,
        seed: int,
        condition: str,
        checkpoint: str,
        episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        logps: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Column arrays of length ``N`` describing the last transition.

        One entry per agent for the tick whose actions were just applied:
        ``t``, ``day`` and ``t_day`` are those of the decision, ``action`` is
        the action taken then, and the remaining fields are the resulting
        state after the tick. Keys follow ``config.LOG_COLUMNS`` exactly, so
        an evaluation loop can collect one dict per tick and concatenate each
        column before building a DataFrame. ``logps`` is ``(N, N_ACTIONS)``
        or ``None`` (NaN columns).
        """
        n = self.cfg.N
        actions = np.asarray(actions, dtype=np.int64).reshape(n)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(n)
        if logps is None:
            logps = np.full((n, N_ACTIONS), np.nan, dtype=np.float32)
        logps = np.asarray(logps, dtype=np.float32).reshape(n, N_ACTIONS)
        cols: dict[str, np.ndarray] = {
            "seed": np.full(n, seed, dtype=np.int64),
            "condition": np.full(n, condition, dtype=object),
            "checkpoint": np.full(n, checkpoint, dtype=object),
            "episode": np.full(n, episode, dtype=np.int64),
            "t": np.full(n, self._step_t, dtype=np.int64),
            "day": np.full(n, self._step_day, dtype=np.int64),
            "t_day": np.full(n, self._step_t_day, dtype=np.int64),
            "agent": self._arange.astype(np.int64),
            "x": self.pos[:, 0].astype(np.int64),
            "y": self.pos[:, 1].astype(np.int64),
            "zone": self.zone.astype(np.int64),
            # the action actually applied: the held action while a journey is running
            "action": self.prev_action.astype(np.int64),
            "decision": self.decision.copy(),
            "active": self.active.copy(),
            "queued": self.queued.copy(),
            "E": self.E.astype(np.float32),
            "F": self.F.astype(np.float32),
            "C": self.C.astype(np.float32),
            # World v2. F is the clipped sum of the two tanks, so both are logged beside it.
            "fast_tank": self.fast.astype(np.float32),
            "slow_tank": self.slow.astype(np.float32),
            "meal_type": np.asarray(_MEAL_NAMES, dtype=object)[self.meal_type],
            "company_others": self.company_others.astype(np.int64),
            "W": self.W.astype(np.float32),
            "M": self.M.astype(np.float32),
            "D": self.D.astype(np.float32),
            "reward": rewards,
            "effort_penalty": self.effort_penalty.astype(np.float32),
            "rest_idle_cost": self.rest_idle_cost.astype(np.float32),
            "coins_spent_food": self.coins_spent_food.astype(np.float32),
            "coins_spent_drink": self.coins_spent_drink.astype(np.float32),
            "drank": self.drank.copy(),
            "informed": self.informed.copy(),
        }
        for k in range(N_ACTIONS):
            cols[f"logp{k}"] = logps[:, k].copy()
        assert list(cols) == LOG_COLUMNS
        return cols

    def log_rows(
        self,
        seed: int,
        condition: str,
        checkpoint: str,
        episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        logps: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """One dict per agent for the last transition; see :meth:`log_columns`."""
        cols = self.log_columns(seed, condition, checkpoint, episode, actions, rewards, logps)
        keys = list(cols)
        rows = []
        for i in range(self.cfg.N):
            rows.append({k: cols[k][i].item() if hasattr(cols[k][i], "item") else cols[k][i] for k in keys})
        return rows
