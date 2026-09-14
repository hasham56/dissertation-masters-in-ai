"""Tests for the hand-written policies and the RLModule wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from hamlet.config import (
    CANTEEN,
    FARM,
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
    GREEDY_WORK_ABOVE,
    HOME,
    IDLE,
    JOBS,
    MARKET,
    N_ACTIONS,
    OFFICE,
    SOCIAL,
    ZONE_RECTS,
    HamletConfig,
)
from hamlet.core import HamletCore
from hamlet.policies import (
    GreedyClockPolicy,
    GreedyClockWorkPolicy,
    GreedyStatePolicy,
    RandomPolicy,
    job_choice,
)

JOB_ACTIONS = (GO_FARM, GO_OFFICE)


# Rules that sit behind flags: by default opening hours are on the jobs only, and the
# two-tank economy is off by default. A test covering one of them switches it on.
ZONE_HOURS = dict(market_open=(70, 190), canteen_open=(110, 230), bar_open=(120, 240))
TWO_TANK = dict(two_tank_economy=True, **ZONE_HOURS)


def daytime_core(**kw) -> HamletCore:
    """A core at an hour when the market, Food Street and the bar are all open, and it is not night.

    World v2 gave Food Street and the bar their own hours, so an hour that only guarantees the
    market is no longer enough to test a scheduler's choice between needs: a shut zone falls
    through to the next-largest deficit and the test would be measuring the clock, not the ranking.
    """
    cfg = HamletConfig(**kw)
    core = HamletCore(cfg, 0)
    core.reset(0)
    # Economy v3 gave the market and Food Street their hours back, so the hour is derived from the
    # config rather than written down: the latest of the four opening ticks, plus a tick so that a
    # zone opening exactly then is already serving. Deriving it means a later change to any zone's
    # hours moves this hour with it instead of silently landing the tests in a shut village.
    core.t = max(cfg.job_open[0], cfg.market_open[0], cfg.canteen_open[0], cfg.bar_open[0]) + 1
    assert cfg.market_is_open(core.t_day) and cfg.canteen_is_open(core.t_day)
    assert cfg.bar_is_open(core.t_day) and cfg.job_is_open(core.t_day)
    assert not cfg.is_night(core.t_day)
    return core


def test_random_policy_shapes():
    core = HamletCore(HamletConfig(), 0)
    obs = core.reset(0)
    pol = RandomPolicy(np.random.default_rng(0))
    a, lp = pol.act(obs, core)
    assert a.shape == (core.cfg.N,) and a.dtype == np.int64
    assert a.min() >= 0 and a.max() < N_ACTIONS
    assert lp.shape == (core.cfg.N, N_ACTIONS) and lp.dtype == np.float32
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)


@pytest.mark.parametrize("policy_cls", [GreedyStatePolicy, GreedyClockPolicy])
def test_greedy_picks_largest_weighted_deficit(policy_cls):
    core = daytime_core(n_agents=4)
    # Economy v3 serves a need only once it is below GREEDY_SERVE_TRIGGER (0.5 at laziness 0), so
    # each level meant to be served is set below it. Agent 3 is the control: every need above the
    # trigger, so it idles rather than topping up the largest of several small deficits.
    core.E[:] = (0.2, 0.9, 0.9, 0.85)
    core.F[:] = (0.9, 0.4, 0.9, 0.9)
    core.C[:] = (0.9, 0.9, 0.1, 0.9)
    # World v2 charges for meals, and an agent that cannot pay goes to work instead, which is a
    # different rule and tested on its own below. Here the purse is full so the ranking is what
    # decides.
    core.W[:] = 50.0
    actions, lp = policy_cls().act(core._observe(), core)
    assert lp is None
    # Agent 1's largest deficit is satiety. In job hours economy v3 serves satiety at the farm,
    # which feeds while it pays, so the ranking chose the need and the farm rule chose the zone.
    assert actions.tolist() == [GO_HOME, GO_FARM, GO_SOCIAL, IDLE]


@pytest.mark.parametrize("policy_cls", [GreedyStatePolicy, GreedyClockPolicy])
def test_greedy_moves_toward_the_right_zone(policy_cls):
    core = daytime_core(n_agents=3)
    pol = policy_cls()
    # World v2 derives F from the two tanks, so hold the tanks rather than F. Agent 0 is hungry
    # in its slow half, which is what the market staple fills.
    # v2-lite carries satiety in the slow tank alone and it caps at 1.0, so "full" is 1.0 here.
    # F is derived during a step, so it is set alongside the tanks rather than left stale.
    core.E[:] = 1.0
    core.slow[:] = (0.1, 1.0, 1.0)
    core.fast[:] = 0.0
    core.F[:] = (0.1, 1.0, 1.0)
    core.C[:] = (1.0, 0.2, 1.0)
    core.W[:] = 50.0
    for _ in range(40):
        core.E[:] = 1.0
        core.slow[:] = (0.1, 1.0, 1.0)
        core.fast[:] = 0.0
        core.F[:] = (0.1, 1.0, 1.0)
        core.C[:] = (1.0, 0.2, 1.0)
        a, _ = pol.act(core._observe(), core)
        core.step(a)
    # Economy v3: in job hours a hungry agent below farm_feed_cap eats at the farm rather than at
    # a counter, so the zone hunger sends it to is the farm.
    assert core.zone[0] == FARM, "agent 0 goes where its hunger is served"
    assert core.zone[1] == SOCIAL and core.zone[2] == HOME


def test_greedy_eats_plain_in_the_morning_and_at_food_street_once_it_opens():
    """The two-tank economy's meal rule (flag on): hours and purse decide, not hunger alone.

    Before Food Street opens the only meal is the market stall, which fills the fast tank. Once it
    opens, the clock-aware scheduler prefers it, because the slow tank is what lasts. The
    clock-blind scheduler prefers it all day and will walk to it while it is shut, which is the
    point of the control.
    """
    # farm_feeds is held off: this test is about choosing between the two counters, and in job
    # hours the farm would feed the agent before either counter was reached.
    cfg = HamletConfig(n_agents=2, farm_feeds=False, **TWO_TANK)
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.E[:] = 1.0
    core.C[:] = 1.0
    core.slow[:] = 0.1
    core.fast[:] = 0.1
    core.F[:] = 0.2                              # derived during a step; set it with the tanks
    core.W[:] = 50.0

    core.t = cfg.market_open[0] + 5              # 07:30: market open, Food Street still shut
    assert cfg.market_is_open(core.t_day) and not cfg.canteen_is_open(core.t_day)
    a, _ = GreedyClockPolicy().act(core._observe(), core)
    assert a.tolist() == [GO_MARKET, GO_MARKET], "plain food in the morning"

    core.t = cfg.canteen_open[0] + 5             # 11:30: both open
    # The rule is "top up the lower half, ties to the cheaper counter", and a staple is cheaper
    # than a sitting. So Food Street wins when the fast half is the emptier one, not on a tie.
    core.slow[:] = 0.30
    core.fast[:] = 0.03
    core.F[:] = 0.33                             # below SATIETY_TRIGGER, so a food trip starts
    assert cfg.market_is_open(core.t_day) and cfg.canteen_is_open(core.t_day)
    b, _ = GreedyClockPolicy().act(core._observe(), core)
    assert b.tolist() == [GO_CANTEEN, GO_CANTEEN], "Food Street when the fast half is the low one"


def test_greedy_goes_to_work_when_it_cannot_pay_for_a_meal():
    """The v2 rule that replaced "the schedulers never work": an empty purse sends them to a job."""
    # farm_feeds is held off so that the job in the first half is the empty purse's doing and not
    # the farm's, and so that the second half measures the counter choice rather than the farm.
    cfg = HamletConfig(n_agents=2, farm_feeds=False, **TWO_TANK)
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.t = cfg.canteen_open[0] + 5             # every counter open, jobs open
    core.E[:] = 1.0
    core.C[:] = 1.0
    core.F[:] = 0.2
    core.W[:] = 0.0                              # cannot afford either meal
    for pol in (GreedyStatePolicy(), GreedyClockPolicy()):
        a, _ = pol.act(core._observe(), core)
        assert a.tolist() in ([GO_FARM, GO_FARM], [GO_FARM, GO_OFFICE],
                              [GO_OFFICE, GO_FARM], [GO_OFFICE, GO_OFFICE]), \
            f"{pol.name} should earn before it eats, got {a.tolist()}"
    # With coins it eats instead, so the job is a consequence of the empty purse and nothing else.
    core.W[:] = 50.0
    a, _ = GreedyClockPolicy().act(core._observe(), core)
    assert set(a.tolist()) <= {GO_CANTEEN, GO_MARKET}, \
        "with coins in hand it eats rather than works; which counter is the tie-break's business"


def test_greedy_weights_follow_core_weights():
    core = daytime_core(n_agents=2, social_on=False)
    core.E[:] = 0.4          # below GREEDY_SERVE_TRIGGER, so energy is eligible
    core.F[:] = 1.0
    core.C[:] = 0.0
    a, _ = GreedyStatePolicy().act(core._observe(), core)
    assert a.tolist() == [GO_HOME, GO_HOME]  # social weight is zero, so social is never chosen
    # With every weighted need above the start trigger the agent idles rather than serving the
    # zero-weight social need, which is at zero and would otherwise be the largest deficit.
    core.E[:] = 1.0
    a, _ = GreedyStatePolicy().act(core._observe(), core)
    assert a.tolist() == [IDLE, IDLE]


def _greedy_state_trace(clock_visible: bool) -> np.ndarray:
    cfg = HamletConfig(clock_visible=clock_visible)
    core = HamletCore(cfg, 4)
    obs = core.reset(4)
    pol = GreedyStatePolicy()
    out = []
    for _ in range(500):
        a, _ = pol.act(obs, core)
        obs, _, _ = core.step(a)
        out.append(a)
    return np.stack(out)


def test_greedy_state_never_reads_the_clock():
    assert np.array_equal(_greedy_state_trace(True), _greedy_state_trace(False))
    # Identical states at different hours give identical actions.
    core = daytime_core(n_agents=2)
    core.E[:] = (0.3, 0.8)
    core.F[:] = (0.8, 0.3)
    core.C[:] = 1.0
    core.W[:] = 100.0
    day_actions, _ = GreedyStatePolicy().act(core._observe(), core)
    core.t = 0  # midnight, market closed
    night_actions, _ = GreedyStatePolicy().act(core._observe(), core)
    assert np.array_equal(day_actions, night_actions)


def test_greedy_clock_rests_at_night_when_tired():
    cfg = HamletConfig(n_agents=3, **ZONE_HOURS)
    core = HamletCore(cfg, 0)
    core.reset(0)
    # World v2: at tick 0 nothing is open, so the contrast would be lost. Ticks 220-229 are both
    # night and Food Street's last hour, which is exactly where the night rule can be told apart
    # from simple hunger.
    core.t = cfg.night[0] + 5
    assert cfg.is_night(core.t_day) and cfg.canteen_is_open(core.t_day)
    assert not cfg.market_is_open(core.t_day), "so the counter left open is the canteen"
    core.E[:] = (GREEDY_NIGHT_REST - 0.05, GREEDY_NIGHT_REST + 0.05, 0.95)
    core.F[:] = (0.1, 0.1, 0.1)
    core.C[:] = 1.0
    # Economy v3 charges at both counters, and at this hour the market has shut, so the purse only
    # has to cover a Food Street sitting for the counter to be a live choice.
    core.W[:] = cfg.food_street_price + 1.0
    a, _ = GreedyClockPolicy().act(core._observe(), core)
    assert a.tolist() == [GO_HOME, GO_CANTEEN, GO_CANTEEN], "only the tired one is sent to bed"
    b, _ = GreedyStatePolicy().act(core._observe(), core)
    # The clock-blind scheduler does not know the jobs shut at 19:00, and in economy v3 the farm
    # feeds whoever works it, so its hunger sends it to the farm at any hour. It keeps trying to
    # eat, which is the contrast this test is about; where it goes to eat is the farm rule's doing.
    assert b.tolist() == [GO_FARM, GO_FARM, GO_FARM], "the clock-blind one keeps trying to eat"


def test_greedy_clock_skips_closed_market():
    # farm_feeds is held off so that the clock-blind scheduler's mistake is walking to a shut
    # stall, which is what this test is about, rather than being fed at the farm.
    cfg = HamletConfig(n_agents=2, farm_feeds=False, **ZONE_HOURS)
    core = HamletCore(cfg, 0)
    core.reset(0)
    # 20:00: the market has shut and Food Street is still serving. The clock-aware scheduler falls
    # back to the canteen; the clock-blind one walks to a stall that closed an hour ago.
    core.t = cfg.market_open[1] + 10
    assert not cfg.is_night(core.t_day)
    assert not cfg.market_is_open(core.t_day) and cfg.canteen_is_open(core.t_day)
    core.E[:] = 1.0
    core.C[:] = 1.0
    core.F[:] = 0.1
    core.W[:] = 100.0
    a, _ = GreedyClockPolicy().act(core._observe(), core)
    assert a.tolist() == [GO_CANTEEN, GO_CANTEEN], "the clock-aware one takes the counter still open"
    b, _ = GreedyStatePolicy().act(core._observe(), core)
    assert b.tolist() == [GO_MARKET, GO_MARKET], "the clock-blind one does not know it is shut"


def _episode_stats(policy, seed: int = 0, threshold: float = 0.3) -> dict[str, float]:
    """Need statistics of one default episode after burn-in.

    ``frac_mean_ok``: fraction of ticks on which every population-mean need is
    above ``threshold``; ``frac_agent_ok``: fraction of agent-ticks on which
    every need of that agent is above it; ``mean_D``: mean drive.
    """
    cfg = HamletConfig()
    core = HamletCore(cfg, seed)
    obs = core.reset(seed)
    mean_ok, agent_ok, drive = [], [], []
    for _ in range(cfg.episode_ticks):
        a, _ = policy.act(obs, core)
        obs, _, _ = core.step(a)
        if core.day >= cfg.burn_in_days:
            levels = np.stack([core.E, core.F, core.C])
            mean_ok.append(bool((levels.mean(axis=1) > threshold).all()))
            agent_ok.append((levels > threshold).all(axis=0))
            drive.append(core.D)
    return {
        "frac_mean_ok": float(np.mean(mean_ok)),
        "frac_agent_ok": float(np.concatenate(agent_ok).mean()),
        "mean_D": float(np.concatenate(drive).mean()),
    }


def test_greedy_clock_keeps_mean_needs_above_threshold():
    """Sanity check of the world: the reference scheduler keeps the population fed, rested and social."""
    stats = _episode_stats(GreedyClockPolicy())
    assert stats["frac_mean_ok"] > 0.9, stats


GATE_SEEDS = (0, 1, 10000, 10001, 10002)


def test_greedy_clock_competence_gate():
    """The competence gate: at least 0.90 of agent-ticks with every need above 0.3.

    The threshold was fixed against baseline runs and is frozen; see
    scripts/report_gate.py for the calibration evidence. The
    gate is the minimum over seeds, not the mean.
    """
    gates = {seed: _episode_stats(GreedyClockPolicy(), seed=seed)["frac_agent_ok"] for seed in GATE_SEEDS}
    assert min(gates.values()) >= 0.90, gates


def _job_trace(policy, seed: int = 0) -> np.ndarray:
    cfg = HamletConfig()
    core = HamletCore(cfg, seed)
    obs = core.reset(seed)
    out = []
    for _ in range(cfg.episode_ticks):
        a, _ = policy.act(obs, core)
        obs, _, _ = core.step(a)
        out.append(a)
    return np.stack(out)


@pytest.mark.parametrize("policy_cls", [GreedyStatePolicy, GreedyClockPolicy])
def test_greedy_schedulers_work_only_for_money_or_food(policy_cls):
    """World v2 overturned "the schedulers never work": they work for a reason, never for its own sake.

    Three rules can send a scheduler to a job. Two are about money: it cannot afford either
    counter, or its purse is below a day's eating times the budget margin during job hours. The
    third arrived with economy v3 and is about food: the farm feeds whoever works it, so a hungry
    agent is sent to the farm and eats while it earns. Every job tick in an episode must be covered
    by one of the three at the moment the action was chosen; a job tick with a full purse, no
    budget shortfall and no hunger would mean the scheduler had started working for its own sake,
    which is not a rule it has.
    """
    cfg = HamletConfig(**TWO_TANK)
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    pol = policy_cls()
    job_ticks, justified = 0, 0
    budget = cfg.daily_budget * cfg.budget_margin
    for _ in range(cfg.episode_ticks):
        a, _ = pol.act(obs, core)
        money = (core.W < min(cfg.market_price, cfg.food_street_price)) | (core.W < budget)
        # The farm-feeds rule only ever points at the farm, and only for an agent whose satiety is
        # below the start trigger, so it justifies nothing the money rules do not already cover at
        # the office.
        food = bool(cfg.farm_feeds) & (core.F < GREEDY_SERVE_TRIGGER) & (a == GO_FARM)
        reason = money | food
        took_job = np.isin(a, JOB_ACTIONS)
        job_ticks += int(took_job.sum())
        justified += int((took_job & reason).sum())
        obs, _, _ = core.step(a)
    assert job_ticks == justified, (
        f"{pol.name} took {job_ticks - justified} job tick(s) it had no reason for")


def test_greedy_clock_work_trigger():
    """The work rule fires by day, jobs open, when the lowest need is above GREEDY_WORK_ABOVE
    and no committed need is still being served; otherwise it is GREEDY-CLOCK."""
    cfg = HamletConfig(n_agents=4)
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.t = cfg.job_open[0]
    assert cfg.job_is_open(core.t_day) and not cfg.is_night(core.t_day)
    # Economy v3's money rule sends any scheduler to a job while its purse is below a day's eating
    # times the margin, so the purse is held above the budget throughout: what is being measured
    # here is GREEDY_WORK_ABOVE and nothing else.
    core.W[:] = cfg.daily_budget * cfg.budget_margin + 1.0
    lo = GREEDY_WORK_ABOVE - 0.05
    hi = GREEDY_WORK_ABOVE + 0.05
    # Agent 2's need below the threshold is social rather than satiety. In job hours economy v3
    # serves satiety at the farm, which is itself a job action, so a hungry agent could not tell
    # the work rule from the farm rule; a social need is served at the bar either way.
    core.E[:] = (1.0, lo, 1.0, hi)
    core.F[:] = (1.0, 1.0, 1.0, hi)
    core.C[:] = (1.0, 1.0, lo, hi)
    pol = GreedyClockWorkPolicy()
    a, lp = pol.act(core._observe(), core)
    assert lp is None
    assert a[0] in JOB_ACTIONS and a[3] in JOB_ACTIONS      # lowest need above the threshold
    assert a[1] == GO_HOME and a[2] == GO_SOCIAL              # one need below it
    base, _ = GreedyClockPolicy().act(core._observe(), core)
    assert base[1] == a[1] and base[2] == a[2]
    assert not np.isin(base, JOB_ACTIONS).any()
    # A committed need is served to the commit level before the rule can fire:
    # agent 1 rests; raising its energy to 0.7 (above the work threshold, below 0.9) keeps it resting.
    core.t += 1
    core.E[1] = 0.7
    a2, _ = pol.act(core._observe(), core)
    assert a2[1] == GO_HOME and a2[0] in JOB_ACTIONS
    core.t += 1
    core.E[1] = GREEDY_COMMIT_UNTIL              # served: the rule may fire now
    a3, _ = pol.act(core._observe(), core)
    assert a3[1] in JOB_ACTIONS
    # A working agent stops when any need falls below the threshold.
    core.t += 1
    core.C[0] = lo
    a4, _ = pol.act(core._observe(), core)
    assert a4[0] == GO_SOCIAL
    # Jobs closed (before 08:00) or night: identical to GREEDY-CLOCK.
    for t in (cfg.job_open[0] - 1, cfg.job_open[1], 0):
        core.t = t
        work, _ = GreedyClockWorkPolicy().act(core._observe(), core)
        plain, _ = GreedyClockPolicy().act(core._observe(), core)
        assert np.array_equal(work, plain), t
        assert not np.isin(work, JOB_ACTIONS).any()


def test_greedy_clock_work_splits_identical_agents_afresh_each_episode():
    """The H3 positive control needs its own scheduler not to carry identity.

    On the neutral population nothing distinguishes the agents, so which job each takes must be a
    property of the episode, not of the agent. On `aptitude_split` it must be a property of the
    agent, because that is the signal the control is built to detect.
    """
    import dataclasses

    base = HamletConfig()
    neutral = [_job_ranking(base, seed) for seed in (10_000, 10_001, 10_002)]
    assert not all(np.array_equal(neutral[0], r) for r in neutral[1:]), \
        "the neutral split repeats across episodes, so the scheduler is carrying identity again"

    # One trait assignment for every evaluation seed, as the positive control does, so that a
    # difference between seeds can only come from the tie-break.
    split = dataclasses.replace(base, population="aptitude_split")
    apt = [_preferred_job(split, seed, traits_seed=0) for seed in (10_000, 10_001, 10_002)]
    assert all(np.array_equal(apt[0], r) for r in apt[1:]), \
        "aptitude no longer decides the job, so the positive control has no signal to detect"


def _job_ranking(cfg: HamletConfig, seed: int) -> np.ndarray:
    """The tie-break the policy draws for one episode."""
    core = HamletCore(cfg, seed)
    core.reset(seed)
    pol = GreedyClockWorkPolicy()
    pol.act(core._observe(), core)
    return pol._tie_break.copy()


def _preferred_job(cfg: HamletConfig, seed: int, traits_seed: int | None = None) -> np.ndarray:
    """The job each agent chooses when both have seats, aptitude included."""
    core = HamletCore(cfg, seed, traits_seed=traits_seed)
    core.reset(seed)
    core.t = cfg.job_open[0]
    core.E[:] = 1.0
    core.F[:] = 1.0
    core.C[:] = 1.0
    core.active[:] = False
    pol = GreedyClockWorkPolicy()
    pol.act(core._observe(), core)
    return job_choice(core, None, pol._tie_break)


def test_greedy_clock_work_ignores_position_and_breaks_ties_by_the_tie_break():
    """The job choice is aptitude, then the tie-break. Nothing spatial.

    Distance used to rank the jobs, which in a world with no distance amounted to ranking them by
    the agent's home and so by its index; see `job_choice`. The contract now is: equal aptitude and
    no tie-break means every agent takes the same job, the default index order putting FARM first;
    a tie-break reorders them per agent; aptitude overrides both; capacity still redirects.
    """
    cfg = HamletConfig(n_agents=4)
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.t = cfg.job_open[0]
    core.E[:] = 1.0
    core.F[:] = 1.0
    core.C[:] = 1.0
    fx0, fy0, fx1, fy1 = ZONE_RECTS[FARM]
    ox0, oy0, ox1, oy1 = ZONE_RECTS[OFFICE]

    # Positions as different as the grid allows; with equal aptitude they change nothing.
    core.pos[0] = (fx0, fy0)
    core.pos[1] = (ox0, oy0)
    core.pos[2] = (0, 0)
    core.pos[3] = core.home[3]
    a = job_choice(core)
    assert a.tolist() == [GO_FARM] * 4
    core.pos[:] = (ox0, oy0)                         # everyone on the office doorstep
    assert job_choice(core).tolist() == a.tolist()   # still FARM: position is not read

    # An explicit tie-break ranks the jobs per agent; 0 is the preferred rank.
    tie = np.array([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=float)
    assert job_choice(core, None, tie).tolist() == [GO_FARM, GO_OFFICE, GO_OFFICE, GO_FARM]

    # The policy draws its own, once per episode, and holds it for the episode.
    pol = GreedyClockWorkPolicy()
    first, _ = pol.act(core._observe(), core)
    drawn = pol._tie_break.copy()
    for _ in range(3):
        pol.act(core._observe(), core)
    assert np.array_equal(pol._tie_break, drawn)
    assert np.array_equal(first, job_choice(core, None, drawn))
    # and it is reproducible from the episode seed
    core_b = HamletCore(cfg, 0)
    core_b.reset(0)
    pol_b = GreedyClockWorkPolicy()
    pol_b.act(core_b._observe(), core_b)
    assert np.array_equal(pol_b._tie_break, drawn)

    # A full job sends the next agent to the other one.
    cfg2 = HamletConfig(n_agents=3, cap_job=2)
    core2 = HamletCore(cfg2, 0)
    core2.reset(0)
    core2.t = cfg2.job_open[0]
    core2.E[:] = 1.0
    core2.F[:] = 1.0
    core2.C[:] = 1.0
    core2.pos[:] = (ox0, oy0)
    core2.active[:] = (True, True, False)
    core2.zone[:] = (OFFICE, OFFICE, OFFICE)
    a2 = job_choice(core2)
    assert a2[2] == GO_FARM
    core2.active[:] = False
    assert job_choice(core2)[2] == GO_FARM           # both free: the default order takes FARM

    # Aptitude overrides the tie-break, whichever way the tie-break points.
    apt = np.ones((4, len(JOBS)))
    apt[1, 1] = 1.5
    against = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    a3 = job_choice(core, apt, against)
    assert a3[1] == GO_OFFICE                        # higher OFFICE aptitude beats a FARM-first tie-break
    assert a3.tolist() == [GO_OFFICE, GO_OFFICE, GO_OFFICE, GO_FARM]
    pol2 = GreedyClockWorkPolicy(aptitude=apt)
    actions, _ = pol2.act(core._observe(), core)
    assert actions[1] == GO_OFFICE
    with pytest.raises(ValueError):
        GreedyClockWorkPolicy(aptitude=np.ones((2, 2))).act(core._observe(), core)

def test_greedy_clock_work_actually_works():
    """On the default world the work rule fires: the fourth baseline spends more time at a job.

    Economy v3's money rule sends every scheduler to a job sometimes, so "GREEDY-CLOCK never
    works" is no longer the claim. What the work rule adds is work the money rules do not ask for,
    and that shows as a strictly larger share of job ticks.
    """
    work = _job_trace(GreedyClockWorkPolicy())
    plain = _job_trace(GreedyClockPolicy())
    assert np.isin(work, JOB_ACTIONS).mean() > 0.01
    assert np.isin(work, JOB_ACTIONS).mean() > np.isin(plain, JOB_ACTIONS).mean()


def test_greedy_clock_night_hysteresis():
    """Once sent home at night the agent stays until the night ends, whatever its energy."""
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    # Economy v3 gave the counters their hours back, so 00:30 has nothing open and the contrast
    # would be lost. Ticks 220-229 are both night and Food Street's last hour, which is where the
    # night rule can be told apart from hunger. The jobs have shut, so the farm cannot feed either.
    core.t = cfg.night[0] + 5
    assert cfg.is_night(core.t_day) and cfg.canteen_is_open(core.t_day)
    assert not cfg.job_is_open(core.t_day)
    pol = GreedyClockPolicy()
    core.E[:] = GREEDY_NIGHT_REST - 0.05
    core.F[:] = 0.05  # very hungry: without the night rule this agent would go to Food Street
    core.W[:] = cfg.food_street_price + 1.0
    core.C[:] = 1.0
    a, _ = pol.act(core._observe(), core)
    assert a.tolist() == [GO_HOME, GO_HOME]
    core.E[:] = 1.0  # fully rested, still night, still hungry
    a, _ = pol.act(core._observe(), core)
    assert a.tolist() == [GO_HOME, GO_HOME]
    literal = GreedyClockPolicy(night_rule="literal")
    a, _ = literal.act(core._observe(), core)
    assert a.tolist() == [GO_CANTEEN, GO_CANTEEN]
    woken = GreedyClockPolicy(night_rule="hysteresis_wake")
    core.E[:] = GREEDY_NIGHT_REST - 0.05
    woken.act(core._observe(), core)
    core.E[:] = 1.0
    a, _ = woken.act(core._observe(), core)
    assert a.tolist() == [GO_CANTEEN, GO_CANTEEN]


def test_greedy_commitment_matters():
    """Without the thermostat hysteresis the scheduler oscillates and the needs collapse."""
    assert GreedyClockPolicy().commit_until == GREEDY_COMMIT_UNTIL
    with_commit = _episode_stats(GreedyClockPolicy())
    without = _episode_stats(GreedyClockPolicy(commit_until=None))
    assert without["frac_mean_ok"] < 0.5 < with_commit["frac_mean_ok"], (without, with_commit)
    assert without["mean_D"] > with_commit["mean_D"]


def test_greedy_commitment_memory_resets_with_the_episode():
    cfg = HamletConfig(n_agents=3)
    pol = GreedyClockPolicy()
    core = HamletCore(cfg, 1)
    obs = core.reset(1)
    for _ in range(50):
        a, _ = pol.act(obs, core)
        obs, _, _ = core.step(a)
    assert pol._committed is not None
    fresh = HamletCore(cfg, 1)
    first, _ = pol.act(fresh.reset(1), fresh)
    again, _ = GreedyClockPolicy().act(fresh.reset(1), fresh)
    assert np.array_equal(first, again)
