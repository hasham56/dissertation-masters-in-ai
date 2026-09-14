"""Unit tests for the simulation core."""
from __future__ import annotations

import numpy as np
import pytest

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
    IDLE,
    JOBS,
    LOG_COLUMNS,
    MARKET,
    N_ACTIONS,
    OBS,
    OBS_FIXED,
    OFFICE,
    PUBLIC_ZONES,
    SOCIAL,
    TRANSIT,
    ZONE_NAMES,
    ZONE_RECTS,
    HamletConfig,
)
from hamlet.core import MEAL, REST, SOCIALISE, HamletCore
from hamlet.policies import RandomPolicy


# Rules that sit behind flags: opening hours are on the jobs only by default, and
# the two-tank economy is off by default. A test covering one of them switches it on and asserts
# what it always asserted.
ZONE_HOURS = dict(market_open=(70, 190), canteen_open=(110, 230), bar_open=(120, 240))
TWO_TANK = dict(two_tank_economy=True, **ZONE_HOURS)


def run_until(core: HamletCore, actions: np.ndarray, pred, limit: int = 200) -> int:
    """Step with constant actions until ``pred(core)`` holds; return ticks used."""
    for k in range(limit):
        if pred(core):
            return k
        core.step(actions)
    raise AssertionError("condition not reached within limit")


def at_zone(core: HamletCore, zone: int) -> bool:
    return bool((core.zone == zone).all())


# ---------------------------------------------------------------- shapes


def test_shapes_and_dtypes():
    cfg = HamletConfig()
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert obs.shape == (cfg.N, cfg.obs_dim) and obs.dtype == np.float32
    obs, rew, info = core.step(np.zeros(cfg.N, dtype=np.int64))
    assert obs.shape == (cfg.N, cfg.obs_dim) and obs.dtype == np.float32
    assert rew.shape == (cfg.N,) and rew.dtype == np.float32
    assert set(info) == {"t", "day", "t_day", "truncated"}
    assert core.pos.shape == (cfg.N, 2) and core.home.shape == (cfg.N, 2)
    assert core.ticks_since.shape == (cfg.N, 3)
    for name in ("E", "F", "C", "W", "M", "D", "coins_earned"):
        assert getattr(core, name).shape == (cfg.N,)
    for name in ("active", "queued", "informed"):
        assert getattr(core, name).dtype == bool
    assert core.prev_action.dtype == np.int64


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "E"])
@pytest.mark.parametrize("symmetry", ["S0", "S1"])
@pytest.mark.parametrize("n", [2, 5, 16])
def test_obs_dim_identical_across_arms(arm, symmetry, n):
    cfg = HamletConfig(arm=arm, symmetry=symmetry, n_agents=n)
    core = HamletCore(cfg, 1)
    obs = core.reset(1)
    assert obs.shape == (n, OBS_FIXED + n)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_obs_and_states_stay_in_unit_interval():
    cfg = HamletConfig()
    core = HamletCore(cfg, 3)
    pol = RandomPolicy(np.random.default_rng(3))
    obs = core.reset(3)
    for _ in range(600):
        a, _ = pol.act(obs, core)
        obs, _, _ = core.step(a)
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        for s in (core.E, core.F, core.C, core.M):
            assert s.min() >= 0.0 and s.max() <= 1.0


def test_truncation_at_episode_ticks():
    cfg = HamletConfig(n_days=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    for k in range(cfg.episode_ticks):
        _, _, info = core.step(np.full(cfg.N, IDLE))
        assert info["truncated"] == (k == cfg.episode_ticks - 1)
        assert info["t"] == k + 1
    with pytest.raises(RuntimeError):
        core.step(np.full(cfg.N, IDLE))


def test_reset_layout():
    cfg = HamletConfig()
    core = HamletCore(cfg, 5)
    core.reset(5)
    assert core.t == 0
    assert (core.pos == core.home).all()
    assert (core.zone == HOME).all()
    expected = {(cfg.home_x, 2 + 2 * k) for k in range(cfg.N)}
    assert {tuple(h) for h in core.home} == expected
    big = HamletCore(HamletConfig(n_agents=16), 0)
    assert len({tuple(h) for h in big.home}) == 16
    assert (big.home < cfg.grid).all()
    for x, y in big.home:
        for x0, y0, x1, y1 in ZONE_RECTS.values():
            assert not (x0 <= x <= x1 and y0 <= y <= y1)
    assert (big._zone_grid[big.home[:, 0], big.home[:, 1]] == TRANSIT).all()
    assert (core.prev_action == -1).all()
    assert core.informed.sum() == 1
    assert (core.W == cfg.init_wealth).all() and (core.M == cfg.init_mood).all()
    for s in (core.E, core.F, core.C):
        assert (s >= cfg.init_state_low).all() and (s <= cfg.init_state_high).all()
    core.reset(5, init_range=(0.1, 0.2))
    for s in (core.E, core.F, core.C):
        assert (s >= 0.1).all() and (s <= 0.2).all()


# ------------------------------------------------------------ movement


def test_a_journey_costs_travel_ticks_then_arrives():
    """Travel is a fixed cost: travel_ticks in TRANSIT, then the arrival tick at the destination.

    Replaces the Manhattan movement test; position is no longer part of the simulation.
    """
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    assert (core.zone == HOME).all() and core.can_decide.all()

    zones = []
    for _ in range(cfg.travel_ticks + 2):
        core.step(np.full(2, GO_FARM))
        zones.append(core.zone.copy())
    # the decision tick and the travel_ticks - 1 that follow it are TRANSIT
    for k in range(cfg.travel_ticks):
        assert (zones[k] == TRANSIT).all(), f"tick {k} should still be in transit"
    assert (zones[cfg.travel_ticks] == FARM).all(), "arrival is at t + travel_ticks"
    assert (zones[cfg.travel_ticks + 1] == FARM).all()



def test_decision_ticks_are_false_during_a_journey_and_true_at_a_zone():
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    flags = []
    for _ in range(cfg.travel_ticks + 3):
        core.step(np.full(2, GO_CANTEEN))
        flags.append(bool(core.decision[0]))
    assert flags[0] is True, "the tick that starts the journey is a decision tick"
    assert not any(flags[1 : cfg.travel_ticks + 1]), "no decision while travelling or on arrival"
    assert flags[cfg.travel_ticks + 1] is True, "the tick after arrival decides again"


def test_the_action_is_held_during_a_journey():
    """An action passed mid-journey is ignored; the journey completes to its original target."""
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.step(np.full(2, GO_FARM))                 # starts a journey to FARM
    for _ in range(cfg.travel_ticks - 1):
        core.step(np.full(2, GO_SOCIAL))           # ignored
    core.step(np.full(2, GO_SOCIAL))               # arrival tick, still ignored
    assert (core.zone == FARM).all()


def test_idle_and_same_zone_requests_are_one_tick():
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.step(np.full(2, IDLE))
    assert (core.zone == HOME).all() and core.decision.all()
    core.step(np.full(2, GO_HOME))                 # already HOME: one tick, and it grants the slot
    assert (core.zone == HOME).all() and core.active.all() and core.decision.all()


def test_go_home_targets_own_tile_and_transit_elsewhere():
    cfg = HamletConfig(n_agents=3)
    core = HamletCore(cfg, 2)
    core.reset(2)
    run_until(core, np.full(3, GO_CANTEEN), lambda c: at_zone(c, CANTEEN))
    run_until(core, np.full(3, GO_HOME), lambda c: at_zone(c, HOME))
    assert (core.pos == core.home).all()
    assert core.active.all()
    # One tile away from home is TRANSIT, also when standing on another home tile.
    core.pos[0] = core.home[1]
    assert core._zone_of(core.pos)[0] == TRANSIT


# ------------------------------------------------------------ capacity


def test_capacity_never_exceeded_and_queued_flagged():
    cfg = HamletConfig(n_agents=8)
    core = HamletCore(cfg, 0)
    core.reset(0)
    cap = cfg.capacity(CANTEEN)
    # World v2: Food Street has hours, and it evicts at close. Start well inside them so the
    # twenty ticks below all fall while it is open.
    run_until(core, np.full(8, GO_CANTEEN),
              lambda c: at_zone(c, CANTEEN) and cfg.canteen_is_open(c.t_day)
                                 and cfg.canteen_is_open(c.t_day + 20))
    for _ in range(20):
        core.step(np.full(8, GO_CANTEEN))
        assert core.active.sum() == cap
        assert core.queued.sum() == 8 - cap
        assert not (core.active & core.queued).any()


def test_slot_holders_keep_slots_and_admission_uses_rng():
    cfg = HamletConfig(n_agents=8)
    core = HamletCore(cfg, 0)
    core.reset(0)
    # World v2: start well inside Food Street's hours so the whole sequence runs while it is open.
    run_until(core, np.full(8, GO_CANTEEN),
              lambda c: at_zone(c, CANTEEN) and cfg.canteen_is_open(c.t_day)
                                 and cfg.canteen_is_open(c.t_day + 20))
    core.step(np.full(8, GO_CANTEEN))
    holders = core.active.copy()
    for _ in range(10):
        core.step(np.full(8, GO_CANTEEN))
        assert (core.active == holders).all()
    # When a holder leaves, exactly one newcomer is admitted.
    leaver = int(np.flatnonzero(holders)[0])
    actions = np.full(8, GO_CANTEEN)
    actions[leaver] = GO_HOME
    core.step(actions)
    assert not core.active[leaver]
    assert core.active.sum() == cfg.capacity(CANTEEN)
    assert core.active[np.flatnonzero(holders)[1]]
    # Admission order comes from core.rng: different seeds, different admissions.
    admitted = set()
    for seed in range(6):
        c = HamletCore(cfg, seed)
        c.reset(seed)
        run_until(c, np.full(8, GO_CANTEEN),
                  lambda cc: at_zone(cc, CANTEEN) and cfg.canteen_is_open(cc.t_day))
        c.step(np.full(8, GO_CANTEEN))
        admitted.add(tuple(np.flatnonzero(c.active)))
    assert len(admitted) > 1


# ---------------------------------------------------------- opening hours


JOB_ACTIONS = {FARM: GO_FARM, OFFICE: GO_OFFICE}


@pytest.mark.parametrize("job", JOBS)
def test_closed_job_grants_nothing(job):
    cfg = HamletConfig(n_agents=2, cap_job=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    go = np.full(2, JOB_ACTIONS[job])
    assert not cfg.job_is_open(core.t_day)
    run_until(core, go, lambda c: at_zone(c, job))
    w0 = core.W.copy()
    while not cfg.job_is_open(core.t_day):
        core.step(go)
        assert not core.active.any() and not core.queued.any()
        assert (core.coins_earned == 0).all()
    assert (core.W == w0).all()
    core.step(go)
    assert core.active.all()
    assert (core.coins_earned > 0).all()
    assert (core.W > w0).all()


@pytest.mark.parametrize("job", JOBS)
def test_job_drains_extra_energy_and_pays_productivity(job):
    cfg = HamletConfig(n_agents=2, cap_job=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    go = np.full(2, JOB_ACTIONS[job])
    run_until(core, go, lambda c: at_zone(c, job))
    run_until(core, np.full(2, IDLE), lambda c: cfg.job_is_open(c.t_day))
    core.E[:] = (cfg.productivity_floor / 2, 1.0)
    core.F[:] = 1.0
    e0 = core.E.copy()
    core.step(go)
    # Economy v3: farm work is heavier, so its work drain carries farm_energy_mult, and the farm
    # pays farm_wage_share of the office wage. Both jobs still pay productivity times their wage.
    mult = cfg.farm_energy_mult if job == FARM else 1.0
    wage = cfg.farm_wage if job == FARM else cfg.office_wage
    assert np.allclose(core.E, e0 - cfg.energy_drain - cfg.energy_work_drain * mult)
    assert np.allclose(core.coins_earned, wage * np.array([0.5, 1.0]))


def test_two_jobs_same_hours_different_wage():
    """FARM and OFFICE open and close on the same ticks; economy v3 pays them differently.

    The office pays ``office_wage`` and the farm ``farm_wage_share`` of it, which is what makes
    the office worth walking to for an agent that does not need feeding. The hours stay shared, so
    the wage is the only thing separating the two jobs.
    """
    cfg = HamletConfig(n_agents=2, cap_job=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    actions = np.array([GO_FARM, GO_OFFICE])
    run_until(core, actions, lambda c: c.zone[0] == FARM and c.zone[1] == OFFICE)
    paid_ticks = {FARM: set(), OFFICE: set()}
    while core.t < cfg.ticks_per_day:
        core.E[:] = 1.0
        core.F[:] = 1.0
        core.step(actions)
        assert core.active[0] == core.active[1]
        assert np.isclose(core.coins_earned[0], core.coins_earned[1] * cfg.farm_wage_share)
        if core.active[0]:
            paid_ticks[FARM].add(core._step_t_day)
            paid_ticks[OFFICE].add(core._step_t_day)
            # World v2 keeps the seat over the unpaid lunch hour but pays nothing for it, so the
            # wage is asserted only on the ticks the job actually pays.
            if cfg.job_pays(core._step_t_day):
                assert np.allclose(core.coins_earned, (cfg.farm_wage, cfg.office_wage))
            else:
                assert np.allclose(core.coins_earned, 0.0)
    assert paid_ticks[FARM] == paid_ticks[OFFICE] == set(range(cfg.job_open[0], cfg.job_open[1]))


def test_job_capacities_are_independent():
    """Each job holds ceil(N/2) workers; filling one does not take slots from the other."""
    cfg = HamletConfig(n_agents=8)
    core = HamletCore(cfg, 0)
    core.reset(0)
    assert cfg.capacity(FARM) == cfg.capacity(OFFICE) == 4
    actions = np.array([GO_FARM] * 6 + [GO_OFFICE] * 2)
    run_until(core, actions, lambda c: (c.zone == np.where(actions == GO_FARM, FARM, OFFICE)).all())
    run_until(core, np.full(8, IDLE), lambda c: cfg.job_is_open(c.t_day))
    core.step(actions)
    assert core.active[:6].sum() == 4 and core.queued[:6].sum() == 2
    assert core.active[6:].all() and not core.queued[6:].any()
    assert (core.coins_earned[core.active] > 0).all() and (core.coins_earned[~core.active] == 0).all()
    obs = core._observe()
    occ = obs[0, OBS["occ"]]
    assert occ[PUBLIC_ZONES.index(FARM)] == pytest.approx(1.0)
    assert occ[PUBLIC_ZONES.index(OFFICE)] == pytest.approx(0.5)


# ------------------------------------------------------------- market


def test_market_purchase_arithmetic():
    cfg = HamletConfig(n_agents=2, init_wealth=25.0, **TWO_TANK)
    core = HamletCore(cfg, 0)
    core.reset(0)
    run_until(core, np.full(2, GO_MARKET), lambda c: at_zone(c, MARKET))
    run_until(core, np.full(2, IDLE), lambda c: cfg.market_is_open(c.t_day))
    # World v2: a market meal is instant and fills the fast tank, and is refused outright when
    # the agent cannot pay. Satiety is the clipped sum of the two tanks, so the arithmetic is
    # checked on the tank the meal touches.
    core.slow[:] = (0.05, 0.09)
    core.fast[:] = 0.0
    core.W[:] = (25.0, cfg.market_price - 0.01)
    f0 = core.slow.copy()
    core.step(np.full(2, GO_MARKET))
    assert core.active.all()
    assert core.purchased[0] and not core.purchased[1]
    assert np.isclose(core.W[0], 25.0 - cfg.market_price)
    assert np.isclose(core.W[1], cfg.market_price - 0.01), "a refused meal debits nothing"
    assert np.isclose(core.slow[0], cfg.tank_cap - cfg.slow_tank_drain), "filled to its cap"
    assert np.isclose(core.slow[1], f0[1] - cfg.slow_tank_drain)  # could not afford
    # Overshoot above 1 is wasted.
    # Overshoot above the tank's cap is wasted, and the meal is still charged in full.
    core.W[:] = 25.0
    core.slow[:] = cfg.tank_cap - 0.01
    core.step(np.full(2, GO_MARKET))
    assert np.allclose(core.slow, cfg.tank_cap - cfg.slow_tank_drain)
    assert core.ticks_since[0, MEAL] == 0


def test_market_closed_grants_nothing():
    cfg = HamletConfig(n_agents=2, init_wealth=25.0, **ZONE_HOURS)
    core = HamletCore(cfg, 0)
    core.reset(0)
    assert not cfg.market_is_open(core.t_day)
    run_until(core, np.full(2, GO_MARKET), lambda c: at_zone(c, MARKET))
    core.step(np.full(2, GO_MARKET))
    assert not core.active.any() and not core.purchased.any()
    assert (core.W == 25.0).all()


# ---------------------------------------------------------------- rest


def test_rest_and_night_bonus():
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    assert cfg.is_night(core.t_day)
    core.E[:] = 0.5
    core.step(np.full(2, GO_HOME))
    assert core.active.all()
    night_gain = cfg.energy_rest_gain * (1 + cfg.night_rest_bonus) - cfg.energy_drain
    assert np.allclose(core.E, 0.5 + night_gain)
    assert (core.ticks_since[:, REST] == 0).all()
    run_until(core, np.full(2, GO_HOME), lambda c: not cfg.is_night(c.t_day))
    core.E[:] = 0.5
    core.step(np.full(2, GO_HOME))
    assert np.allclose(core.E, 0.5 + cfg.energy_rest_gain - cfg.energy_drain)


def test_canteen_gain_and_meal_counter():
    cfg = HamletConfig(n_agents=2, cap_canteen=2, **TWO_TANK)
    core = HamletCore(cfg, 0)
    core.reset(0)
    run_until(core, np.full(2, GO_CANTEEN), lambda c: at_zone(c, CANTEEN))
    # World v2 gave Food Street opening hours, so wait for them before eating.
    run_until(core, np.full(2, GO_CANTEEN), lambda c: cfg.canteen_is_open(c.t_day))
    core.fast[:] = 0.05
    core.slow[:] = 0.0
    w0 = core.W.copy()
    core._seated[:] = False
    core.step(np.full(2, GO_CANTEEN))
    assert np.allclose(core.fast, 0.05 + cfg.fast_tank_fill - cfg.fast_tank_drain)
    assert np.allclose(core.W, w0 - cfg.food_street_price), "the sitting is charged once"
    w1 = core.W.copy()
    core.step(np.full(2, GO_CANTEEN))
    assert np.allclose(core.W, w1), "and not again while the agent stays seated"
    assert (core.ticks_since[:, MEAL] == 0).all()
    assert (core.ticks_since[:, REST] > 0).all()


# -------------------------------------------------------------- social


def test_social_gain_scales_with_company():
    """World v2 replaced v1's all-or-nothing rule: alone earns half, company earns more.

    The bar keeps a fixed staff presence, so a solo drinker is not in an empty room and receives
    bar_solo_share of the gain. The floor and the slope are covered in full by
    tests/test_world_v2.py; this checks the same arithmetic through the core the rest of this
    file exercises.
    """
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    run_until(core, np.full(2, GO_SOCIAL), lambda c: at_zone(c, SOCIAL) and cfg.bar_is_open(c.t_day))
    core.C[:] = 0.5
    # Alone: agent 0 at the bar, agent 1 idle (present but not active).
    core.step(np.array([GO_SOCIAL, IDLE]))
    assert core.active[0] and not core.active[1]
    assert core.company_others[0] == 0
    assert np.isclose(core.C[0], 0.5 - cfg.social_drain + cfg.social_gain * cfg.bar_solo_share)
    assert core.ticks_since[0, SOCIALISE] == 0
    # Together: one companion each, so the multiplier rises by one step.
    core.C[:] = 0.5
    core.step(np.full(2, GO_SOCIAL))
    mult = cfg.bar_solo_share + cfg.bar_company_step * 1
    assert np.allclose(core.C, 0.5 + cfg.social_gain * mult - cfg.social_drain)


def test_social_off_zeroes_weight_and_obs():
    cfg = HamletConfig(n_agents=2, social_on=False)
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert core.weights[2] == 0.0
    assert (obs[:, 2] == 0).all()
    assert np.allclose(core.D, (1 - core.E) ** 2 + (1 - core.F) ** 2)
    run_until(core, np.full(2, GO_SOCIAL), lambda c: at_zone(c, SOCIAL))
    c0 = core.C.copy()
    obs, _, _ = core.step(np.full(2, GO_SOCIAL))
    assert (obs[:, 2] == 0).all()
    assert np.allclose(core.C, c0 - cfg.social_drain)


def test_gossip_copies_only_on_co_presence():
    cfg = HamletConfig(n_agents=3)
    core = HamletCore(cfg, 0)
    core.reset(0)
    run_until(core, np.full(3, GO_SOCIAL),
              lambda c: at_zone(c, SOCIAL) and cfg.bar_is_open(c.t_day)
                                 and cfg.bar_is_open(c.t_day + 5))
    core.informed[:] = (True, False, False)
    # Informed agent active alone: nothing spreads.
    core.step(np.array([GO_SOCIAL, IDLE, IDLE]))
    assert core.informed.tolist() == [True, False, False]
    # Two uninformed agents together: nothing to copy.
    core.step(np.array([IDLE, GO_SOCIAL, GO_SOCIAL]))
    assert core.informed.tolist() == [True, False, False]
    # Informed with one other: only that one learns.
    core.step(np.array([GO_SOCIAL, GO_SOCIAL, IDLE]))
    assert core.informed.tolist() == [True, True, False]
    cfg2 = HamletConfig(n_agents=3, gossip=False)
    core2 = HamletCore(cfg2, 0)
    core2.reset(0)
    assert not core2.informed.any()


# ------------------------------------------------------------ obs layout


def test_obs_blocks():
    cfg = HamletConfig(n_agents=4)
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert np.allclose(obs[:, OBS["states"]], np.stack([core.E, core.F, core.C], 1), atol=1e-6)
    assert np.allclose(obs[:, OBS["wealth"]], cfg.init_wealth / cfg.wealth_obs_scale)
    assert np.allclose(obs[:, OBS["clock"]], (0.5, 1.0))  # t_day = 0
    # xy, dist and home_dist are zero-filled: position is no longer part of the simulation
    assert (obs[:, OBS["xy"]] == 0).all()
    assert (obs[:, OBS["dist"]] == 0).all()
    assert (obs[:, OBS["home_dist"]] == 0).all()
    assert (obs[:, OBS["zone"]].argmax(1) == HOME).all()
    assert (obs[:, OBS["prev_action"]] == 0).all()
    assert (obs[:, OBS["active"]] == 0).all()
    assert (obs[:, OBS["occ"]] == 0).all()
    # PUBLIC_ZONES order: FARM, OFFICE, CANTEEN, SOCIAL, MARKET. At t_day = 0 the jobs and the
    # market are shut. The observation announces the job and market hours only: Food Street and
    # the bar have hours of their own in economy v3, but the open block never carried them, so
    # their flags read 1 all day and an agent has to learn those hours from experience.
    assert np.allclose(obs[:, OBS["open"]], (0.0, 0.0, 1.0, 1.0, 0.0))
    assert not cfg.canteen_is_open(0) and not cfg.market_is_open(0)
    assert np.allclose(obs[:, OBS["agent_id"]], np.eye(4))
    obs, _, _ = core.step(np.full(4, GO_HOME))
    assert (obs[:, OBS["prev_action"]].argmax(1) == GO_HOME).all()
    assert (obs[:, OBS["active"]] == 1).all()
    # The spatial entries stay zero whatever the agent does, and the layout is unchanged.
    assert (obs[:, OBS["xy"]] == 0).all() and (obs[:, OBS["dist"]] == 0).all()
    assert (obs[:, OBS["home_dist"]] == 0).all()
    assert obs.shape[1] == cfg.obs_dim


def test_obs_layout_widths():
    """The fixed layout follows the zone and action counts: five public zones, seven zones, seven actions."""
    assert len(ZONE_NAMES) == 7 and N_ACTIONS == 7 and len(ACTION_NAMES) == 7
    assert PUBLIC_ZONES == (FARM, OFFICE, CANTEEN, SOCIAL, MARKET)
    assert JOBS == (FARM, OFFICE)
    widths = {k: (v.stop - v.start) for k, v in OBS.items() if isinstance(v, slice) and v.stop is not None}
    # World v2 inserts the two hunger tanks at 3 and 4, so everything after them shifts by two
    # and the fixed block is 48 rather than 46.
    assert widths == {"states": 3, "tanks": 2, "clock": 2, "xy": 2, "dist": 5, "occ": 5, "open": 5,
                      "zone": 7, "prev_action": 7, "traits": 7}
    assert OBS["wealth"] == 5 and OBS["home_dist"] == 25 and OBS["active"] == 40
    assert OBS["zone"] == slice(26, 33) and OBS["prev_action"] == slice(33, 40)
    assert OBS["traits"] == slice(41, 48)
    assert OBS["agent_id"] == slice(OBS_FIXED, None) and OBS_FIXED == 48
    assert HamletConfig(n_agents=8).obs_dim == 56
    assert [c for c in LOG_COLUMNS if c.startswith("logp")] == [f"logp{k}" for k in range(7)]
    # Every zone and every action is reachable through the one-hot blocks.
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    for action in range(N_ACTIONS):
        # let any running journey finish first: prev_action carries the held action during one
        while not core.can_decide.all():
            core.step(np.full(2, IDLE))
        obs, _, _ = core.step(np.full(2, action))
        assert (obs[:, OBS["prev_action"]].argmax(1) == action).all()
        assert obs[:, OBS["prev_action"]].sum() == 2
    core.reset(0)
    run_until(core, np.full(2, GO_OFFICE), lambda c: at_zone(c, OFFICE))
    obs = core._observe()
    assert (obs[:, OBS["zone"]].argmax(1) == OFFICE).all()


def test_zone_rectangles_do_not_overlap_homes_or_each_other():
    cfg = HamletConfig(n_agents=16)
    cfg.validate()
    rects = list(ZONE_RECTS.values())
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]
    for x, y in cfg.home_tiles():
        for x0, y0, x1, y1 in rects:
            assert not (x0 <= x <= x1 and y0 <= y <= y1)


def test_occupancy_fraction():
    cfg = HamletConfig(n_agents=8)
    core = HamletCore(cfg, 0)
    core.reset(0)
    # World v2 gave Food Street and the bar opening hours; occupancy is only meaningful while open.
    run_until(core, np.full(8, GO_CANTEEN), lambda c: at_zone(c, CANTEEN) and cfg.canteen_is_open(c.t_day))
    obs, _, _ = core.step(np.full(8, GO_CANTEEN))
    assert np.allclose(obs[:, OBS["occ"].start + PUBLIC_ZONES.index(CANTEEN)], 1.0)
    run_until(core, np.full(8, GO_SOCIAL), lambda c: at_zone(c, SOCIAL) and cfg.bar_is_open(c.t_day))
    obs, _, _ = core.step(np.full(8, GO_SOCIAL))
    assert np.allclose(obs[:, OBS["occ"].start + PUBLIC_ZONES.index(SOCIAL)], 1.0)
    obs, _, _ = core.step(np.array([GO_SOCIAL] * 4 + [IDLE] * 4))
    assert np.allclose(obs[:, OBS["occ"].start + PUBLIC_ZONES.index(SOCIAL)], 0.5)


@pytest.mark.parametrize("arm", ["B", "D"])
def test_hidden_arms_zero_fill_states(arm):
    cfg = HamletConfig(arm=arm)
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert (obs[:, OBS["states"]] == 0).all()
    pol = RandomPolicy(np.random.default_rng(0))
    for _ in range(50):
        a, _ = pol.act(obs, core)
        obs, _, _ = core.step(a)
    assert (obs[:, OBS["states"]] == 0).all()
    assert (obs[:, OBS["wealth"]] > 0).any()


def test_arm_e_writes_elapsed_time_proxies():
    cfg = HamletConfig(arm="E", n_agents=2)
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert (obs[:, OBS["states"]] == 0).all()
    for _ in range(10):
        obs, _, _ = core.step(np.full(2, IDLE))
    assert np.allclose(obs[:, OBS["states"]], 10 / cfg.proxy_scale)
    core.ticks_since[:] = 10 * int(cfg.proxy_scale)
    obs = core._observe()
    assert (obs[:, OBS["states"]] == 1.0).all()


def test_symmetry_s0_zeroes_id_block():
    cfg = HamletConfig(symmetry="S0")
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert (obs[:, OBS["agent_id"]] == 0).all()
    assert obs.shape[1] == cfg.obs_dim


def test_clock_hidden_zero_fills_but_keeps_open_flags():
    cfg = HamletConfig(clock_visible=False)
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    assert (obs[:, OBS["clock"]] == 0).all()
    run_until(core, np.full(cfg.N, IDLE), lambda c: cfg.job_is_open(c.t_day))
    obs = core._observe()
    assert (obs[:, OBS["clock"]] == 0).all()
    assert obs[0, OBS["open"].start + PUBLIC_ZONES.index(FARM)] == 1.0
    assert obs[0, OBS["open"].start + PUBLIC_ZONES.index(OFFICE)] == 1.0


# -------------------------------------------------------------- reward


def test_level_reward_is_minus_drive():
    cfg = HamletConfig(n_agents=3)
    core = HamletCore(cfg, 0)
    core.reset(0)
    _, rew, _ = core.step(np.full(3, IDLE))
    assert np.allclose(rew, -core.D, atol=1e-6)
    w = core.weights
    D = w[0] * (1 - core.E) ** 2 + w[1] * (1 - core.F) ** 2 + w[2] * (1 - core.C) ** 2
    assert np.allclose(core.D, D)


def test_difference_reward():
    cfg = HamletConfig(n_agents=3, reward_form="difference")
    core = HamletCore(cfg, 0)
    core.reset(0)
    d_old = core.D.copy()
    _, rew, _ = core.step(np.full(3, GO_HOME))
    assert np.allclose(rew, cfg.difference_scale * (d_old - core.D), atol=1e-5)


@pytest.mark.parametrize("arm", ["C", "D"])
def test_income_reward(arm):
    cfg = HamletConfig(n_agents=2, arm=arm, income_scale=2.0, cap_job=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    _, rew, _ = core.step(np.full(2, IDLE))
    assert (rew == 0).all()
    actions = np.array([GO_FARM, GO_OFFICE])
    run_until(core, actions, lambda c: c.zone[0] == FARM and c.zone[1] == OFFICE)
    run_until(core, np.full(2, IDLE), lambda c: cfg.job_is_open(c.t_day))
    _, rew, _ = core.step(actions)
    assert np.allclose(rew, cfg.income_scale * core.coins_earned)
    assert (rew > 0).all()


def test_mood_ema():
    cfg = HamletConfig(n_agents=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    m0 = core.M.copy()
    core.step(np.full(2, IDLE))
    assert np.allclose(core.M, cfg.mood_tau * m0 + (1 - cfg.mood_tau) * (1 - core.D / 3))


# -------------------------------------------------------------- shocks


def test_jobs_closed_shock_closes_both_jobs():
    cfg = HamletConfig(n_agents=2, shock="jobs_closed_d4", cap_job=2)
    assert cfg.condition_name.endswith("jobs_closed_d4")
    core = HamletCore(cfg, 0)
    core.reset(0)
    actions = np.array([GO_FARM, GO_OFFICE])
    run_until(core, actions, lambda c: c.zone[0] == FARM and c.zone[1] == OFFICE)
    earned_by_day = {FARM: {}, OFFICE: {}}
    while core.t < cfg.episode_ticks:
        core.E[:] = 1.0  # keep productivity at one so only the shock can zero income
        core.F[:] = 1.0
        core.step(actions)
        for k, job in enumerate((FARM, OFFICE)):
            earned_by_day[job][core._step_day] = earned_by_day[job].get(core._step_day, 0.0) + core.coins_earned[k]
    for job in (FARM, OFFICE):
        assert earned_by_day[job][cfg.shock_day] == 0.0
        for day in range(cfg.n_days):
            if day != cfg.shock_day:
                assert earned_by_day[job][day] > 0.0


def test_energy_shock():
    cfg = HamletConfig(n_agents=2, shock="energy_x2_d4")
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.E[:] = 0.9
    core.step(np.full(2, IDLE))
    assert np.allclose(core.E, 0.9 - cfg.energy_drain)
    core.t = cfg.shock_day * cfg.ticks_per_day
    core.E[:] = 0.9
    core.step(np.full(2, IDLE))
    assert np.allclose(core.E, 0.9 - cfg.shock_energy_mult * cfg.energy_drain)


# ----------------------------------------------------- reproducibility


def _trajectory(seed: int, policy_seed: int, ticks: int = 300) -> np.ndarray:
    cfg = HamletConfig()
    core = HamletCore(cfg, seed)
    obs = core.reset(seed)
    pol = RandomPolicy(np.random.default_rng(policy_seed))
    out = []
    for _ in range(ticks):
        a, _ = pol.act(obs, core)
        obs, rew, _ = core.step(a)
        out.append(np.concatenate([obs.ravel(), rew, core.pos.ravel(), core.active]))
    return np.stack(out)


def test_same_seed_same_trajectory():
    a = _trajectory(11, 5)
    b = _trajectory(11, 5)
    assert np.array_equal(a, b)
    c = _trajectory(12, 5)
    assert not np.array_equal(a, c)


def test_construction_equals_reset_with_seed():
    cfg = HamletConfig()
    core = HamletCore(cfg, 9)
    home, E = core.home.copy(), core.E.copy()
    core.reset(9)
    assert np.array_equal(home, core.home) and np.array_equal(E, core.E)
    core.reset()
    assert not (np.array_equal(home, core.home) and np.array_equal(E, core.E))


# -------------------------------------------------------------- logging


def test_log_columns_and_rows():
    cfg = HamletConfig(n_agents=3)
    core = HamletCore(cfg, 0)
    core.reset(0)
    actions = np.array([GO_HOME, GO_FARM, IDLE])
    _, rew, _ = core.step(actions)
    cols = core.log_columns(1, "A-S1-N3", "none", 0, actions, rew, None)
    assert list(cols) == LOG_COLUMNS
    assert all(v.shape == (3,) for v in cols.values())
    assert (cols["t"] == 0).all() and (cols["agent"] == np.arange(3)).all()
    assert np.isnan(cols["logp0"]).all()
    assert (cols["action"] == actions).all()
    rows = core.log_rows(1, "A-S1-N3", "none", 0, actions, rew, np.zeros((3, N_ACTIONS), np.float32))
    assert len(rows) == 3 and list(rows[0]) == LOG_COLUMNS
    assert rows[1]["action"] == GO_FARM and rows[0]["zone"] == HOME
    assert rows[2]["logp6"] == 0.0
    _, rew, _ = core.step(actions)
    assert core.log_columns(1, "x", "none", 0, actions, rew)["t"][0] == 1
