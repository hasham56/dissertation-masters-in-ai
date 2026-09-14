"""Policies that drive :class:`hamlet.core.HamletCore`.

Every policy exposes ``name`` and ``act(obs, core) -> (actions, log_probs)``
where ``actions`` is ``int64 (N,)`` and ``log_probs`` is ``float32
(N, N_ACTIONS)`` or ``None`` when the policy has no distribution to report.

The hand-written baselines read the true internal states from the core, not
from the observation, so they behave identically whether or not an arm hides
those states. They are reference points for the learned agents, not
competitors under the same information constraints. Three of them never
choose a job (GREEDY-STATE and GREEDY-CLOCK serve needs only, RANDOM has no
preference); GREEDY-CLOCK-WORK adds one rule that sends an agent with nothing
to serve to a job.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from hamlet.config import (
    CANTEEN,
    IDLE,
    GO_CANTEEN,
    GO_FARM,
    GO_HOME,
    GO_MARKET,
    GO_OFFICE,
    GO_SOCIAL,
    GREEDY_COMMIT_UNTIL,
    GREEDY_MARKET_HUNGER,
    GREEDY_NIGHT_REST,
    GREEDY_SERVE_TRIGGER,
    GREEDY_NIGHT_RULE,
    GREEDY_NIGHT_RULES,
    GREEDY_WORK_ABOVE,
    HOME,
    JOBS,
    MARKET,
    N_ACTIONS,
    SOCIAL,
)
from hamlet.core import HamletCore


class Policy(Protocol):
    """Structural type of anything that can act in the core."""

    name: str

    def act(self, obs: np.ndarray, core: HamletCore) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(actions int64 (N,), log_probs float32 (N, N_ACTIONS) or None)``."""
        ...


class RandomPolicy:
    """Uniform over the ``N_ACTIONS`` actions; log-probabilities are ``log(1/N_ACTIONS)``."""

    name = "RANDOM"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, obs: np.ndarray, core: HamletCore) -> tuple[np.ndarray, np.ndarray]:
        n = obs.shape[0]
        actions = self.rng.integers(0, N_ACTIONS, size=n, dtype=np.int64)
        log_probs = np.full((n, N_ACTIONS), np.log(1.0 / N_ACTIONS), dtype=np.float32)
        return actions, log_probs

# World v2: a food trip starts only when satiety falls below this, and then runs until the half
# being filled is full. Each half caps at cfg.tank_cap = 0.5, so satiety reaches 1.0 only when both
# are full.
#
# 0.40, not 0.5. At 0.5 the trigger sat exactly at what one full half
# delivers, so an agent that filled its slow half was fed and hungry at the same instant and
# re-evaluated every tick for ever: it churned cheap staples and never saved for a sitting. The
# trigger has to sit below what one half can carry for the hysteresis to have any room.
# Not lower than 0.40 either: the walk to a counter is travel_ticks = 9 ticks at up to
# fast_tank_drain + slow_tank_drain = 0.004 a tick, so leaving at 0.40 arrives at about 0.364 and
# stays the right side of the competence gate's 0.3 floor.
SATIETY_TRIGGER = 0.40



def _greedy_actions(
    core: HamletCore,
    use_clock: bool,
    committed: Optional[np.ndarray] = None,
    commit_until: Optional[float] = None,
    sleeping: Optional[np.ndarray] = None,
    night_rule: str = GREEDY_NIGHT_RULE,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared body of the greedy schedulers.

    Weighted deficits ``w * (1 - level)`` for energy, satiety and social are
    compared and the largest wins (ties go to the earlier need, so energy
    first). Energy sends the agent home, satiety to the canteen (or to the
    market when it can afford a meal and satiety is below
    ``GREEDY_MARKET_HUNGER``), social to the social square.

    A need is eligible for service only while its level is below the agent's
    start trigger, ``GREEDY_SERVE_TRIGGER - greedy_threshold_drop * laziness``
    (the laziness trait, read from ``core.traits``); a need with zero weight
    is never eligible. An agent with no eligible need IDLEs (stays on its
    tile). On the defaults that is 0.5 at laziness 0 falling to 0.2 at
    laziness 1, so a lazy agent starts serving a need later and spends more of
    the day where it stands. Economy v3 put the trigger here; before it the
    ceiling was ``greedy_base_threshold`` (1.0), so every level short of full
    counted as a deficit and the scheduler nibbled. ``greedy_base_threshold``
    no longer gates anything and is kept only so an archived config still
    loads.

    With ``use_clock`` the scheduler also knows the time of day: a closed
    zone is never chosen (its need falls through to the next-largest deficit,
    and the closed market gives way to the canteen), and during the agent's
    own night window (``core.night_window()``, the night shifted by the
    chronotype trait) an agent with energy below ``GREEDY_NIGHT_REST`` goes
    home. Under the ``"hysteresis"`` night rule (the definition of
    GREEDY-CLOCK) the agent then stays home until its window ends;
    ``sleeping`` (``bool (N,)``) carries that state between ticks and is
    updated in place. ``"literal"`` re-evaluates the threshold every tick and
    ``"hysteresis_wake"`` releases the agent once its energy is full; both
    are kept only for the calibration report. Without ``use_clock`` neither
    the clock nor the chronotype is read.

    ``committed`` (``int (N,)``, ``-1`` for none) and ``commit_until`` give
    the scheduler its thermostat hysteresis: an agent keeps serving its
    committed need while that need's level is below ``commit_until`` and its
    zone is open (the act threshold does not release a commitment). With
    ``commit_until=None`` the choice is re-evaluated from scratch every tick,
    which makes the agents oscillate between zones and spend most of the day
    in transit.

    Returns ``(actions int64 (N,), need int64 (N,))`` where ``need`` is the
    index (0 energy, 1 satiety, 2 social) the action serves, ``-1`` for an
    agent that IDLEs.
    """
    cfg = core.cfg
    n = cfg.N
    rows = np.arange(n)
    levels = np.stack([core.E, core.F, core.C], axis=1)  # (N, 3)
    deficits = core.weights[None, :] * (1.0 - levels)
    # A need starts being served only once it is below the start trigger, and the commitment below
    # then carries it to commit_until. Laziness lowers the trigger, so a lazy agent starts later.
    threshold = GREEDY_SERVE_TRIGGER - cfg.greedy_threshold_drop * core.traits.laziness
    eligible = (levels < threshold[:, None]) & (core.weights[None, :] > 0.0)
    # Satiety has its own hysteresis: a food trip starts only when satiety is below
    # SATIETY_TRIGGER, and once started it runs until the half being filled reaches its cap. Without
    # it the scheduler nibbles, and a Food Street sitting is charged in full every time it sits back
    # down. The laziness threshold does not apply: eating is not an errand an idle agent skips.
    # v2-lite keeps v1's threshold-and-commitment behaviour for satiety; the two-tank world uses
    # the SATIETY_TRIGGER hysteresis, which only makes sense when a half can cap out.
    if cfg.two_tank_economy:
        eligible[:, 1] = (core.F < SATIETY_TRIGGER) & (core.weights[None, 1] > 0.0)

    # World v2: both meals cost coins, and a meal the agent cannot pay for is refused outright,
    # so the scheduler must check the purse before it walks anywhere. Food Street fills the slow
    # tank and is preferred whenever it is open and affordable; the market stall is the plain
    # morning meal that fills the fast tank. An agent that can afford neither is `broke_for_meal`
    # and is sent to work instead, below.
    # Rule (a): never travel to a counter the agent cannot pay at. Affordability is checked before
    # the walk, not on arrival, so a refused meal is never the reason for a wasted journey.
    # Rule (c): among the counters it can afford and reach, take the cheapest.
    # Rule (i): a counter whose half is already at its cap is not a candidate. Buying food an
    # agent cannot hold is money burned, and it is what made the scheduler buy three staples a day
    # against a slow half that was already full.
    slow_room = cfg.tank_cap - core.slow
    fast_room = cfg.tank_cap - core.fast
    if not cfg.two_tank_economy:
        # v2-lite: one satiety level carried in the slow tank, and the canteen is free, so it is
        # always a candidate while it is open and satiety is not full.
        slow_room = 1.0 - core.slow
        fast_room = np.zeros(n)
        market_useful = slow_room > 1e-9
        canteen_useful = market_useful
        # A sitting is charged once, so an agent already at a counter needs nothing more.
        can_market = ((core.W >= cfg.market_price) | core._at_market) & market_useful
        can_canteen = (cfg.free_canteen | (core.W >= cfg.food_street_price) | core._seated) & canteen_useful
    else:
        market_useful = slow_room > 1e-9
        canteen_useful = fast_room > 1e-9
        can_market = (core.W >= cfg.market_price) & market_useful
    # A sitting is charged once, when the seat is taken, so an agent already seated needs nothing
    # more; one about to sit down needs the full price.
        can_canteen = ((core.W >= cfg.food_street_price) | core._seated) & canteen_useful
    if use_clock:
        canteen_ok = can_canteen & cfg.canteen_is_open(core.t_day)
        market_ok = can_market & cfg.market_is_open(core.t_day)
    else:
        canteen_ok = can_canteen & np.ones(n, dtype=bool)
        market_ok = can_market & np.ones(n, dtype=bool)
    # Each tank holds half of satiety, so the counters are not substitutes and "cheapest" is only
    # the tie-break. Serving satiety means topping up the half that is actually low: the market
    # staple fills the slow half, a Food Street sitting the fast half. An agent whose slow tank is
    # emptier goes to the stall, one whose fast tank is emptier sits down, and where both are
    # equally low the cheaper counter wins.
    # The food decision, evaluated whenever satiety is below SATIETY_TRIGGER:
    #   candidates = counters that are open, whose half is below its cap, and whose price the purse
    #                covers. canteen_ok and market_ok are exactly that.
    #   if any candidate: take the one whose half is lower, ties to the cheaper counter.
    #   elif a job is open: work (rule b).
    #   else: satiety yields and the next need in deficit order is served.
    # The choice is made among candidates only. Picking the lower half first and then asking
    # whether it was affordable is what sent agents to a counter they could not pay at.
    if cfg.two_tank_economy:
        # Two halves: top up whichever is lower, ties to the cheaper counter.
        cheaper_is_canteen = cfg.food_street_price <= cfg.market_price
        prefer_canteen = np.where(fast_room == slow_room, cheaper_is_canteen,
                                  fast_room > slow_room)
    else:
        # Economy v3, rule 2. In job hours the farm feeds while it pays, so a hungry agent eats
        # there and earns at the same time. Outside job hours it buys: the market stall first
        # because it is the cheaper counter, then Food Street, and if neither is open and
        # affordable it does not set out at all.
        prefer_canteen = ~market_ok & canteen_ok
    both = canteen_ok & market_ok
    satiety_action = np.where(
        both, np.where(prefer_canteen, GO_CANTEEN, GO_MARKET),
        np.where(canteen_ok, GO_CANTEEN, GO_MARKET)).astype(np.int64)
    # Rule (ii): rule (b) fires on the counter the agent actually needs. The lower half is the one
    # worth filling, so an agent whose lower half is unaffordable goes to work even if it could
    # have paid at the other counter. Satiety then drops out of the ranking entirely, so the need
    # falls through and the job is chosen; otherwise the agent walks to a counter it cannot pay at
    # and stands there, which is what it did before this rule.
    # No candidate at all: satiety yields this tick, and rule (b) sends the agent to work if a job
    # is open. "No candidate" means every counter is shut, already full, or unaffordable.
    no_meal = ~(canteen_ok | market_ok)
    broke_for_meal = no_meal & (market_useful | canteen_useful)
    # Rule 3, enforced on the main path and not only on the evening promotion: company costs a
    # drink, and an agent that cannot buy one gets nothing at the bar, because no drink means no
    # company. Before this, social could still win the deficit ranking on affordability grounds it
    # had never been asked about: agents walked to the bar broke, sat there gaining nothing, and the
    # commitment held them. That was 40% of all job-hour ticks.
    no_drink = ~((core.W >= cfg.drink_price) | core._drinking)
    need_actions = np.stack(
        [np.full(n, GO_HOME, dtype=np.int64), satiety_action, np.full(n, GO_SOCIAL, dtype=np.int64)],
        axis=1,
    )

    if use_clock:
        # World v2 gave Food Street and the bar their own hours, so both join the closed set.
        zone_open = {
            HOME: True,
            CANTEEN: cfg.canteen_is_open(core.t_day),
            SOCIAL: cfg.bar_is_open(core.t_day),
            MARKET: cfg.market_is_open(core.t_day),
        }
        need_zone = np.stack(
            [np.full(n, HOME), np.where(satiety_action == GO_CANTEEN, CANTEEN, MARKET),
             np.full(n, SOCIAL)], axis=1
        )
        closed = np.zeros_like(deficits, dtype=bool)
        for zone, is_open in zone_open.items():
            if not is_open:
                closed |= need_zone == zone
        deficits = np.where(closed, -np.inf, deficits)

    # Needs are ranked in deficit order only among the options the agent can actually take, so an
    # unaffordable meal never outranks a need it could have served.
    unservable = ((no_meal[:, None] & (np.arange(3)[None, :] == 1))
                  | (no_drink[:, None] & (np.arange(3)[None, :] == 2)))
    deficits = np.where(unservable, -np.inf, deficits)
    choice = np.where(eligible, deficits, -np.inf)
    need = np.argmax(choice, axis=1)
    serving = np.isfinite(choice[rows, need])
    keep = np.zeros(n, dtype=bool)
    if committed is not None and commit_until is not None:
        has = committed >= 0
        idx = np.where(has, committed, 0)
        keep = has & (levels[rows, idx] < commit_until) & np.isfinite(deficits[rows, idx])
        # A satiety commitment runs until the half it is filling is full, not until F reaches
        # commit_until: the two halves cap at cfg.tank_cap each, so F can never reach 0.9 from one.
        if cfg.two_tank_economy:
            half_full = (np.where(satiety_action == GO_CANTEEN, core.fast, core.slow)
                         >= cfg.tank_cap - 1e-9)
            sat_commit = has & (idx == 1) & ~half_full & np.isfinite(deficits[rows, 1])
            keep = np.where(idx == 1, sat_commit, keep)
        need = np.where(keep, idx, need)
        serving = serving | keep
    actions = np.where(serving, need_actions[rows, need], IDLE)
    need = np.where(serving, need, -1)

    # World v2: a hungry agent with no coins earns them first. Work is the only source of coins,
    # so an empty purse makes the job the way to serve satiety rather than a competing errand.
    # Rules (ii) and (iii), both served by the same job action.
    #   (ii) an agent that cannot pay at the counter its lower half needs works instead;
    #   (iii) the budget rule: during job hours, an agent whose purse is below a day's eating
    #         works, whether or not it is hungry yet. This is what turns work from a last resort
    #         into a routine. Without it the scheduler works only when already too poor to eat, so
    #         raising the wage made it work *less*, and a day's productive ticks fell instead of
    #         rising towards the target.
    # Eating still outranks both: an agent past its satiety trigger goes to the counter first.
    # When the farm feeds, a hungry agent turned away from a full canteen works the farm
    # rather than waiting. The canteen comes first because it feeds faster and costs no energy; the
    # office never serves hunger, because it does not feed at all.
    # "Hungry" for the money rule means below the start trigger, not merely below full: a fed
    # agent in job hours should take the office, which pays double, not the farm.
    hungry = core.F < GREEDY_SERVE_TRIGGER
    if cfg.farm_feeds and ((not use_clock) or cfg.job_is_open(core.t_day)):
        # Rule 2, first branch: in job hours a hungry agent eats at the farm while it earns, so it
        # never needs to choose between the two. The farm only feeds up to farm_feed_cap, so an
        # agent already past the cap is not sent there for food: it falls through to the counters,
        # which is the cap's whole purpose.
        farm_can_feed = core.F < cfg.farm_feed_cap
        actions = np.where((need == 1) & farm_can_feed, GO_FARM, actions)

    jobs_available = (not use_clock) or cfg.job_is_open(core.t_day)
    hungry_broke = ((broke_for_meal & eligible[:, 1]) | (no_drink & eligible[:, 2])) & (need != 1)
    # The budget rule funds the two-tank economy. In v2-lite the canteen is free, so there is
    # nothing to save for and an agent sent to work by a budget it never needs would simply be
    # working instead of living.
    # Rule 4, the money rule: in job hours, an agent whose purse is below a day's drink and meal
    # works. Rule 3 decides which job: the farm if it is hungry, since the farm feeds, the office
    # otherwise, since the office pays double.
    below_budget = (core.W < cfg.daily_budget * cfg.budget_margin) & (need != 1) & ~keep
    take_job = hungry_broke | (below_budget if jobs_available else np.zeros(n, dtype=bool))
    if jobs_available and take_job.any():
        # Rule 4: the farm if hungry, because it feeds; the office otherwise, because it pays four
        # times as much. Falling through to job_choice here was a fault: that picks by aptitude and
        # tie-break, so a solvent, well-fed agent went to the farm about half the time and the
        # office was barely used at any wage.
        jb = (np.where(hungry, GO_FARM, GO_OFFICE) if cfg.farm_feeds else job_choice(core))
        actions = np.where(take_job & (jb >= 0), jb, actions)
        need = np.where(take_job & (jb >= 0), -1, need)

    if use_clock:
        # The bar comes before home in the evening: while it is open and the agent's own night has
        # not started, an eligible social deficit outranks rest. Company is only available while
        # the bar is open, and resting is available all night, so serving social first costs the
        # agent nothing it cannot make up later.
        # Only an agent that is not actually tired goes drinking first. Without this the rule
        # would send an exhausted agent to the bar on any social deficit at all, and at laziness 0
        # every level below 1.0 counts as a deficit, so it would fire almost every evening and
        # sink the competence gate. "Before home" means ahead of an ordinary early night, not
        # ahead of sleep the agent needs.
        # A live commitment outranks this. Without that the rule breaks the thermostat: an agent
        # rests to 0.9, is sent to the bar, loses energy walking there, comes back, and thrashes.
        # Measured: allowing the override put GREEDY-CLOCK at 0.467 of its ticks
        # in TRANSIT against 0.26 in the v1 world.
        # Rule 3: company costs a drink. An agent that cannot buy one does not set out for the bar;
        # it earns first, which rule 4 then arranges.
        # Rule 3: the bar is for a social level that has actually fallen, and a drink must be
        # affordable. One visit, one drink, and the commitment holds until social reaches 0.9.
        evening = (bool(cfg.bar_is_open(core.t_day)) & ~core.night_window()
                   & (core.C < GREEDY_SERVE_TRIGGER)
                   & ((core.W >= cfg.drink_price) | core._drinking))
        rested = core.E >= GREEDY_NIGHT_REST
        bar_first = (evening & rested & eligible[:, 2] & np.isfinite(deficits[:, 2])
                     & (need == 0) & ~keep)
        actions = np.where(bar_first, GO_SOCIAL, actions)
        need = np.where(bar_first, 2, need)

        if night_rule not in GREEDY_NIGHT_RULES:
            raise ValueError(f"unknown night rule {night_rule!r}")
        night = core.night_window()
        tired = core.E < GREEDY_NIGHT_REST
        if night_rule == "literal" or sleeping is None:
            go_home = tired & night
        else:
            sleeping &= night          # an agent whose window has ended is released
            sleeping |= tired & night
            if night_rule == "hysteresis_wake":
                sleeping &= core.E < 1.0
            go_home = sleeping
        actions = np.where(go_home, GO_HOME, actions)
    return actions.astype(np.int64), need.astype(np.int64)


class _GreedyBase:
    """State shared by the greedy schedulers: the per-agent commitment memory.

    ``commit_until`` (default ``GREEDY_COMMIT_UNTIL``) is the level at which a
    committed need is considered served; ``None`` switches the hysteresis off.
    The memory is cleared whenever the core is at ``t == 0``, so one policy
    instance can be reused across episodes.
    """

    name = "GREEDY"
    use_clock = False

    def __init__(
        self,
        commit_until: Optional[float] = GREEDY_COMMIT_UNTIL,
        night_rule: str = GREEDY_NIGHT_RULE,
    ) -> None:
        self.commit_until = commit_until
        self.night_rule = night_rule
        self._committed: Optional[np.ndarray] = None
        self._sleeping: Optional[np.ndarray] = None

    def act(self, obs: np.ndarray, core: HamletCore) -> tuple[np.ndarray, None]:
        n = core.cfg.N
        if self._sleeping is None or core.t == 0 or self._sleeping.shape[0] != n:
            self._sleeping = np.zeros(n, dtype=bool)
        if self.commit_until is None:
            actions, _ = _greedy_actions(
                core, self.use_clock, sleeping=self._sleeping, night_rule=self.night_rule
            )
            return self._override(actions, core), None
        if self._committed is None or core.t == 0 or self._committed.shape[0] != n:
            self._committed = np.full(n, -1, dtype=np.int64)
        actions, need = _greedy_actions(
            core, self.use_clock, self._committed, self.commit_until, self._sleeping, self.night_rule
        )
        self._committed = need
        return self._override(actions, core), None

    def _override(self, actions: np.ndarray, core: HamletCore) -> np.ndarray:
        """Hook for subclasses; the plain schedulers change nothing."""
        return actions


class GreedyStatePolicy(_GreedyBase):
    """GREEDY-STATE-v2: clock-blind need-following that has to pay for its meals.

    Every rule, in the order the scheduler applies them:

    - Compare the weighted deficits ``w * (1 - level)`` for energy, satiety and social and serve
      the largest. Ties go to the earlier need, so energy first.
    - A need is eligible only while its level is below the agent's start trigger,
      ``GREEDY_SERVE_TRIGGER - greedy_threshold_drop * laziness``: 0.5 at laziness 0 on the
      defaults, falling to 0.2 at laziness 1, so laziness is what decides how far a need has to
      fall before the agent will leave the house for it. A need with zero weight is never
      eligible, and an agent with no eligible need idles where it stands.
    - Energy sends the agent home. Social sends it to the bar. Satiety sends it to a meal.
    - **Never walk to a counter it cannot pay at.** Affordability is checked before the journey,
      not on arrival, so a refused meal is never the reason for a wasted trip. This is rule (a).
    - **If no affordable counter is reachable and a job is open, go to work.** Work is the only
      source of coins, so an empty purse makes the job the way to serve satiety rather than a
      competing errand. It works at the higher-aptitude job with a free seat, or the other if that
      one is full. This is rule (b).
    - **Otherwise top up the half that is low.** Each tank holds at most ``tank_cap``, half of
      satiety, so the two counters are not substitutes: serving satiety up to ``commit_until``
      means buying a staple at the market when the slow tank is low and taking a Food Street
      sitting when the fast tank is low and it can afford one. Where both halves are equally low
      the cheaper counter wins, which is ``canteen_price`` a tick against ``market_price`` a meal.
      This is rule (c).
    - **Needs are ranked in deficit order only among affordable options.** An unaffordable meal
      never outranks a need the agent could actually have served, so satiety drops out of the
      ranking entirely when nothing is payable.
    - Hold the chosen need until its level reaches ``commit_until``, so the agent does not
      oscillate between zones. A commitment is released only when its level is restored.
    - Read no clock at all: no opening hours, no night, no chronotype. It will walk to a closed
      zone and stand there, which is the point of the control.
    - **The restlessness charge never falls on this scheduler.** ``commit_until`` is
      ``GREEDY_COMMIT_UNTIL`` = 0.9 and ``rest_idle_energy`` is also 0.9, so the agent releases its
      energy commitment and leaves HOME on the very tick the world would start counting idle ticks
      against it. It cannot accumulate the ``rest_idle_after`` free ticks, let alone pay. At night
      the night rule holds it at home, but the charge is off between 22:00 and 06:00, so that costs
      it nothing either. The references therefore pay zero restlessness by construction, which is
      what makes them a clean line for a learned policy that might not.
    """

    name = "GREEDY-STATE"
    use_clock = False


class GreedyClockPolicy(_GreedyBase):
    """GREEDY-CLOCK-v2: the same needs, read against the clock. The reference line.

    Everything GREEDY-STATE-v2 does, and then these, in the order the scheduler applies them:

    - **Never travel to a closed zone.** Every window is read: the jobs 07:00-19:00, the market
      07:00-19:00, Food Street 11:00-23:00 and the bar 12:00-24:00. A need whose zone is shut is
      not served this tick; its deficit falls through to the next largest.
    - **Eat plain in the morning.** Before Food Street opens at 11:00 the only counter serving is
      the market stall, whose meal lands in the slow tank and carries the agent through the day.
      From 11:00 Food Street is also reachable, and the cheapest affordable rule decides between
      them; a Food Street sitting tops up the fast tank, which is the quick fix rather than the
      staple.
    - **Go to the bar before home in the evening.** While the bar is open and the agent's own
      night has not started, an eligible social deficit outranks rest. Company is available only
      while the bar is open and resting is available all night, so serving social first costs the
      agent nothing it cannot make up later. This is the rule that makes the bar reachable at all
      under a scheduler that would otherwise always rest first.
    - **Honour the night rule.** During the agent's own night window, the world's night shifted
      by its chronotype trait, an agent with energy below ``GREEDY_NIGHT_REST`` goes home and
      stays there until the window ends. That hysteresis is the registered definition;
      ``night_rule`` selects one of the rejected alternatives for the calibration report only.
    - The unpaid lunch hour is deliberately not read. The jobs do not evict over it, so an agent
      already at work stays at work and simply earns nothing for that hour, which is what the
      world does to it rather than something it chooses.
    - **The restlessness charge never falls on this scheduler.** ``commit_until`` is
      ``GREEDY_COMMIT_UNTIL`` = 0.9 and ``rest_idle_energy`` is also 0.9, so the agent releases its
      energy commitment and leaves HOME on the very tick the world would start counting idle ticks
      against it. It cannot accumulate the ``rest_idle_after`` free ticks, let alone pay. At night
      the night rule holds it at home, but the charge is off between 22:00 and 06:00, so that costs
      it nothing either. The references therefore pay zero restlessness by construction, which is
      what makes them a clean line for a learned policy that might not.
    """

    name = "GREEDY-CLOCK"
    use_clock = True


def job_choice(core: HamletCore, aptitude: Optional[np.ndarray] = None,
               tie_break: Optional[np.ndarray] = None) -> np.ndarray:
    """Job action per agent (``GO_FARM`` or ``GO_OFFICE``), or ``-1`` when no job has a free seat.

    The preferred job is the one with the higher aptitude (``aptitude`` is
    ``(N, len(JOBS))``; ``None`` reads the aptitude trait from
    ``core.traits``, all ones meaning no preference); ties go to the
    tie-break. If the preferred job has no free seat this tick the other job
    is taken; if neither has one the result is ``-1`` and the caller falls
    back to need-following.

    Ties used to go first to the nearer job, by Manhattan
    distance from the agent's position to the job rectangle. That term was a
    vestige of the grid world: journeys have cost
    a flat ``travel_ticks`` since then and there is no distance to be nearer
    by. It was not inert. Homes are laid out in a column in agent order and no
    longer permuted per episode, so distance to FARM rose and distance to
    OFFICE fell with the agent index, and a term weighted 1.0 against the
    tie-break's 1e-3 silently handed identical agents a fixed, identity-linked
    division of labour. It is dropped; see ``GreedyClockWorkPolicy``.

    ``tie_break`` is ``(N, len(JOBS))`` and ranks the jobs for an agent whose
    aptitude and distance do not separate them; lower is preferred. ``None``
    keeps the historical rank by job index, FARM first, which is the same
    order for every agent and therefore hands identical agents a split fixed
    by agent index. ``GreedyClockWorkPolicy`` passes a per-episode random one
    instead; see its docstring. Aptitude is weighted 1e6 against the
    tie-break's 1e-3, so a tie-break never overrides a real preference.
    """
    cfg = core.cfg
    n = cfg.N
    n_jobs = len(JOBS)
    free = np.empty(n_jobs, dtype=bool)
    for k, job in enumerate(JOBS):
        cap = cfg.capacity(job)
        taken = int((core.active & (core.zone == job)).sum())
        free[k] = cap is None or taken < cap
    apt = core.traits.aptitude if aptitude is None else np.asarray(aptitude, dtype=np.float64).reshape(n, n_jobs)
    # Rank jobs by aptitude, then by the tie-break. Nothing spatial: see the docstring.
    if tie_break is None:
        last = np.broadcast_to(np.arange(n_jobs, dtype=np.float64), (n, n_jobs))
    else:
        last = np.asarray(tie_break, dtype=np.float64).reshape(n, n_jobs)
    score = apt * 1e6 - last * 1e-3
    order = np.argsort(-score, axis=1)                       # (N, n_jobs), best first
    choice = np.full(n, -1, dtype=np.int64)
    for rank in range(n_jobs - 1, -1, -1):
        cand = order[:, rank]
        choice = np.where(free[cand], cand, choice)          # later (better) ranks overwrite
    job_actions = np.array([GO_FARM, GO_OFFICE], dtype=np.int64)
    assert len(job_actions) == n_jobs
    return np.where(choice >= 0, job_actions[np.maximum(choice, 0)], -1)


class GreedyClockWorkPolicy(GreedyClockPolicy):
    """GREEDY-CLOCK with a job-first rule during job hours (one definition for every population).

    During job hours an agent works at the higher-aptitude job with a free
    seat (ties to its own tie-break; the other job if the preferred one is
    full) unless its lowest need is below ``GREEDY_WORK_ABOVE``. Then it fixes that
    need exactly as GREEDY-CLOCK does, serving the largest weighted deficit
    until it reaches ``GREEDY_COMMIT_UNTIL``, and returns to work. A need
    being served is finished before work resumes, so a meal is never
    interrupted. If neither job has a seat the agent follows its needs.
    Outside job hours and inside its own night window it is identical to
    GREEDY-CLOCK. No daily work budget. The job choice uses the aptitude
    trait of ``core.traits``; the explicit ``aptitude`` argument
    (``(N, len(JOBS))``) overrides it, for reports.

    Where aptitude does not separate the two jobs, which is every agent on the
    neutral population, the tie is broken by a per-episode random ranking drawn
    once at ``core.t == 0``. It used to be
    broken by distance from the agent's home to each job rectangle, and homes
    sit in a column in agent order, so the scheduler itself handed identical
    agents a fixed, identity-linked division of labour: agents 0 and 1 took FARM
    and the rest OFFICE in every episode of every seed. That read as a neutral
    cross-episode ARI of 0.84 against the 0.2 the H3 positive control requires
    of a population with no traits. Agents whose aptitudes do differ are
    untouched, because aptitude is weighted 1e6 against the tie-break's 1e-3.

    The draw is seeded from the core's own bit-generator state, read without
    consuming it, so it is reproducible from the episode seed and the world
    stream is left exactly as it was: an episode's weather does not change, only
    which of two identical jobs each agent prefers.

    Coins are never in the drive, so this baseline works because the rule
    says so, not because it values them. It is not an H1 reference: the work
    rule is itself clock-entrained, so its clock gain sits above GREEDY-CLOCK's.
    """

    name = "GREEDY-CLOCK-WORK"
    use_clock = True

    def __init__(
        self,
        aptitude: Optional[np.ndarray] = None,
        work_above: float = GREEDY_WORK_ABOVE,
        commit_until: Optional[float] = GREEDY_COMMIT_UNTIL,
        night_rule: str = GREEDY_NIGHT_RULE,
    ) -> None:
        super().__init__(commit_until=commit_until, night_rule=night_rule)
        self.aptitude = None if aptitude is None else np.asarray(aptitude, dtype=np.float64)
        self.work_above = float(work_above)
        self._working: Optional[np.ndarray] = None
        self._tie_break: Optional[np.ndarray] = None

    @staticmethod
    def _draw_tie_break(core: HamletCore, n: int) -> np.ndarray:
        """One random ranking of the jobs per agent, ``(N, len(JOBS))``, fixed for the episode.

        Seeded from the core's bit-generator state, which is read and not advanced, so the ranking
        follows the episode seed and no world draw moves because of it.
        """
        state = core.rng.bit_generator.state["state"]["state"]
        gen = np.random.default_rng(int(state) % (2 ** 63))
        return np.argsort(gen.random((n, len(JOBS))), axis=1).astype(np.float64)

    def act(self, obs: np.ndarray, core: HamletCore) -> tuple[np.ndarray, None]:
        cfg = core.cfg
        n = cfg.N
        if self._working is None or core.t == 0 or self._working.shape[0] != n:
            self._working = np.zeros(n, dtype=bool)
            self._committed = np.full(n, -1, dtype=np.int64)
            self._sleeping = np.zeros(n, dtype=bool)
            self._tie_break = self._draw_tie_break(core, n)
        if self.aptitude is not None and self.aptitude.shape != (n, len(JOBS)):
            raise ValueError(f"aptitude must be ({n}, {len(JOBS)}), got {self.aptitude.shape}")
        levels = np.stack([core.E, core.F, core.C], axis=1)
        lowest = levels.min(axis=1)
        rows = np.arange(n)
        committed = self._committed if self._committed is not None else np.full(n, -1, dtype=np.int64)
        has = committed >= 0
        unfinished = has & (levels[rows, np.where(has, committed, 0)] < (self.commit_until or 1.0))
        hours = cfg.job_is_open(core.t_day) & ~core.night_window()
        self._working &= (lowest > self.work_above) & hours
        start = hours & (lowest > self.work_above) & ~unfinished & ~self._working
        self._working |= start
        actions, need = _greedy_actions(
            core, True, committed, self.commit_until, self._sleeping, self.night_rule
        )
        job = (job_choice(core, self.aptitude, self._tie_break) if self._working.any()
               else np.full(n, -1, dtype=np.int64))
        take_job = self._working & (job >= 0)
        actions = np.where(take_job, job, actions)
        # a working agent carries no commitment; when it stops, the largest deficit is served afresh
        self._committed = np.where(take_job, -1, need)
        return actions.astype(np.int64), None


class FallbackPolicy:
    """Wraps a checkpoint written by :mod:`hamlet.train_fallback`.

    The actor network is rebuilt from the ``.pt`` file (torch imported
    lazily); actions are sampled from the softmax with ``rng`` by
    :func:`sample_actions`, and the full log-probability vector is returned.
    """

    name = "PPO-FALLBACK"

    def __init__(self, path: Any, rng: Optional[np.random.Generator] = None, name: str = "PPO-FALLBACK") -> None:
        from hamlet.train_fallback import load_actor  # imports torch

        self.actor, self.checkpoint = load_actor(path)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.name = name

    def act(self, obs: np.ndarray, core: HamletCore) -> tuple[np.ndarray, np.ndarray]:
        import torch

        with torch.no_grad():
            logits = self.actor(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        return sample_actions(log_probs, self.rng), log_probs


def sample_actions(log_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one action per row of ``log_probs (N, N_ACTIONS)`` by inverse CDF with ``rng``."""
    probs = np.exp(np.asarray(log_probs, dtype=np.float64))
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random(probs.shape[0])
    cumulative = np.cumsum(probs, axis=1)
    actions = (cumulative < u[:, None]).sum(axis=1)
    return np.minimum(actions, N_ACTIONS - 1).astype(np.int64)
