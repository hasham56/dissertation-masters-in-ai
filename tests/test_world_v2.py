"""World v2: opening hours, eviction, the two hunger tanks, coins, company gain, the sleep bonus.

One test per locked rule of the v2 world. Everything here drives HamletCore
directly rather than through a policy, so a failure names the rule and not the scheduler.

Travel costs ``cfg.travel_ticks`` TRANSIT ticks plus an arrival tick, so a helper walks the agents
to a zone before the rule under test is exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

from hamlet.config import (
    CANTEEN, FARM, GO_CANTEEN, GO_FARM, GO_HOME, GO_MARKET, GO_SOCIAL, HOME, IDLE,
    GO_OFFICE, GREEDY_SERVE_TRIGGER, MARKET, OFFICE, SOCIAL, HamletConfig,
)
from hamlet.core import HamletCore
from hamlet.policies import GreedyClockPolicy


# The rules below sit behind flags: opening hours are on the jobs only by default,
# and the two-tank economy is off by default. Each test switches its own rule back on and asserts
# exactly what it always asserted, so nothing is lost when a default changes.
ZONE_HOURS = dict(market_open=(70, 190), canteen_open=(110, 230), bar_open=(120, 240))
TWO_TANK = dict(two_tank_economy=True, **ZONE_HOURS)


def core(seed: int = 0, **kw) -> HamletCore:
    c = HamletCore(HamletConfig(**kw), seed)
    c.reset(seed)
    return c


def at_tick(c: HamletCore, t_day: int) -> None:
    """Idle the world forward until the next step lands on ``t_day``."""
    n = c.cfg.N
    while c.t_day != t_day:
        c.step(np.full(n, IDLE, dtype=np.int64))


def send(c: HamletCore, action: int, who: np.ndarray | None = None) -> None:
    """Walk every selected agent to ``action``'s zone: travel_ticks transit plus the arrival tick."""
    n = c.cfg.N
    a = np.full(n, IDLE, dtype=np.int64)
    sel = np.ones(n, dtype=bool) if who is None else who
    a[sel] = action
    for _ in range(c.cfg.travel_ticks + 1):
        c.step(a)


# ---------------------------------------------------------------- opening hours
@pytest.mark.parametrize("t_day,expected", [(119, False), (120, True), (239, True)])
def test_bar_hours(t_day, expected):
    """The bar serves 12:00 to midnight, ticks 120 to 239, when bar hours are configured."""
    assert HamletConfig(**ZONE_HOURS).bar_is_open(t_day) is expected


@pytest.mark.parametrize("t_day,expected", [(69, False), (70, True), (189, True), (190, False)])
def test_job_and_market_hours(t_day, expected):
    """Both jobs and the market run 07:00 to 19:00, ticks 70 to 189."""
    cfg = HamletConfig(**ZONE_HOURS)
    assert cfg.job_is_open(t_day) is expected
    assert cfg.market_is_open(t_day) is expected


@pytest.mark.parametrize("t_day,expected", [(109, False), (110, True), (229, True), (230, False)])
def test_food_street_hours(t_day, expected):
    """Food Street serves 11:00 to 23:00, ticks 110 to 229."""
    assert HamletConfig(**ZONE_HOURS).canteen_is_open(t_day) is expected


def test_jobs_do_not_pay_over_lunch_but_stay_open():
    """12:00 to 13:00 pays nothing, and the job stays open so the seat is not lost."""
    cfg = HamletConfig()
    for t in range(120, 130):
        assert cfg.job_is_open(t) is True, "the job is open over lunch"
        assert cfg.job_pays(t) is False, "but it pays nothing"
    assert cfg.job_pays(119) is True and cfg.job_pays(130) is True


# ---------------------------------------------------------------- eviction
def test_food_street_evicts_at_close_and_the_agent_may_decide():
    """At 23:00 whoever is still seated stops being active and is free to choose again."""
    c = core(**ZONE_HOURS)
    at_tick(c, 200)
    send(c, GO_CANTEEN)
    assert c.active.any(), "agents should be seated before close"
    at_tick(c, 229)
    c.step(np.full(c.cfg.N, GO_CANTEEN, dtype=np.int64))    # the last serving tick
    assert c.active.any()
    c.step(np.full(c.cfg.N, GO_CANTEEN, dtype=np.int64))    # 23:00: shut
    assert not c.active.any(), "nobody is active at a closed Food Street"
    assert not c.queued.any(), "and nobody is left queueing"
    assert c.can_decide.all(), "an evicted agent takes a decision tick"


def test_bar_evicts_at_midnight():
    c = core(**ZONE_HOURS)
    at_tick(c, 200)
    send(c, GO_SOCIAL)
    assert c.active.any()
    at_tick(c, 239)
    c.step(np.full(c.cfg.N, GO_SOCIAL, dtype=np.int64))     # the last serving tick
    assert c.active.any()
    c.step(np.full(c.cfg.N, GO_SOCIAL, dtype=np.int64))     # midnight rolls to t_day 0
    assert not c.active.any()
    assert c.can_decide.all()


def test_the_jobs_do_not_evict_over_lunch():
    """An agent at its desk at 12:00 keeps its seat through to 13:00."""
    c = core()
    at_tick(c, 100)
    send(c, GO_FARM)
    seated = c.active.copy()
    assert seated.any()
    at_tick(c, 119)
    for _ in range(10):                                      # straight through the unpaid hour
        c.step(np.full(c.cfg.N, GO_FARM, dtype=np.int64))
        assert (c.active & seated).any(), "the seat is held over lunch"
    assert 120 <= c.t_day <= 130


# ---------------------------------------------------------------- the two tanks
def test_satiety_is_the_clipped_sum_of_the_two_tanks():
    c = core()
    for _ in range(30):
        c.step(np.full(c.cfg.N, IDLE, dtype=np.int64))
        assert np.allclose(c.F, np.minimum(1.0, c.fast + c.slow))
def test_a_market_meal_is_refused_outright_when_the_coins_are_short():
    """No gain, no debit, meal_type none: a refusal leaves no trace but the refusal itself."""
    c = core()
    at_tick(c, 100)
    c.W[:] = c.cfg.market_price - 0.01          # a penny short, every agent
    send(c, GO_MARKET)
    # The fast tank empties in well under a day at the starting constants, so refill it: the point
    # of this test is that a refused meal changes nothing, which needs a tank off its floor.
    c.fast[:] = 0.5
    fast_at_stall, w_at_stall = c.fast.copy(), c.W.copy()
    c.step(np.full(c.cfg.N, GO_MARKET, dtype=np.int64))
    at_stall = c.active & (c.zone == MARKET)
    assert at_stall.any()
    assert np.allclose(c.W[at_stall], w_at_stall[at_stall]), "no coins are taken"
    assert np.all(c.fast[at_stall] < fast_at_stall[at_stall]), "and the tank only drains"
    assert np.all(c.meal_type[at_stall] == 0), "meal_type is none (0 on the core, 'none' in the log)"
    assert c.refused_meal[at_stall].all()


def test_a_food_street_meal_is_refused_when_the_coins_are_short():
    c = core(**TWO_TANK)
    at_tick(c, 150)
    send(c, GO_CANTEEN)
    seated = c.active & (c.zone == CANTEEN)
    assert seated.any()
    # Food Street charges once per sitting. Leaving and coming back is a new sitting, so an agent
    # that cannot cover the price is refused the seat outright.
    c._seated[:] = False
    c.W[:] = c.cfg.food_street_price - 0.01
    fast_before = c.fast.copy()
    c.step(np.full(c.cfg.N, GO_CANTEEN, dtype=np.int64))
    assert np.allclose(c.W[seated], c.cfg.food_street_price - 0.01), "no coins are taken"
    assert np.all(c.fast[seated] <= fast_before[seated]), "and the tank only drains"
    assert np.all(c.meal_type[seated] == 0)


# ---------------------------------------------------------------- company gain
def test_company_gain_floor_and_slope():
    """social_gain x (0.5 + 0.25 x min(others, 2)): 0.5 alone, 0.75 with one, 1.0 with two or more."""
    cfg = HamletConfig()
    for others, expected in ((0, 0.50), (1, 0.75), (2, 1.00), (3, 1.00), (8, 1.00)):
        mult = cfg.bar_solo_share + cfg.bar_company_step * min(others, cfg.bar_company_cap)
        assert mult == pytest.approx(expected), f"{others} other villagers"


def test_a_solo_drinker_still_gains_because_the_staff_are_there():
    """The v1 world gave a solo visit nothing; v2 gives it half, which is the point of the bar."""
    c = core()
    at_tick(c, 150)
    one = np.zeros(c.cfg.N, dtype=bool)
    one[0] = True
    send(c, GO_SOCIAL, one)
    assert c.active[0] and c.zone[0] == SOCIAL
    before = c.C[0]
    a = np.full(c.cfg.N, IDLE, dtype=np.int64)
    a[0] = GO_SOCIAL
    c.step(a)
    cfg = c.cfg
    assert c.company_others[0] == 0, "staff are not villagers and never count as company"
    expected = before - cfg.social_drain + cfg.social_gain * cfg.bar_solo_share
    assert c.C[0] == pytest.approx(min(expected, 1.0))


def test_company_others_counts_villagers_only():
    c = core()
    at_tick(c, 150)
    three = np.zeros(c.cfg.N, dtype=bool)
    three[:3] = True
    send(c, GO_SOCIAL, three)
    a = np.full(c.cfg.N, IDLE, dtype=np.int64)
    a[:3] = GO_SOCIAL
    c.step(a)
    drinking = c.active & (c.zone == SOCIAL)
    assert drinking.sum() == 3
    assert np.all(c.company_others[drinking] == 2), "each sees the other two, and no staff"
    assert np.all(c.company_others[~drinking] == 0)


# ---------------------------------------------------------------- the sleep bonus
def test_the_sleep_bonus_applies_when_the_flag_was_live_on_arrival():
    c = core()
    at_tick(c, 150)
    send(c, GO_SOCIAL)
    c.step(np.full(c.cfg.N, GO_SOCIAL, dtype=np.int64))      # earn company gain, setting the flag
    assert (c._sleep_bonus_until > c.t).any()
    c.E[:] = 0.3                                             # room to rest
    send(c, GO_HOME)                                         # 10 ticks, inside the 60-tick window
    resting = c.active & (c.zone == HOME)
    assert resting.any()
    assert c._sleep_bonus_stay[resting].all(), "the flag was live when they arrived"
    before = c.E.copy()
    c.step(np.full(c.cfg.N, GO_HOME, dtype=np.int64))
    cfg = c.cfg
    gain = cfg.energy_rest_gain * (1.0 + cfg.night_rest_bonus * c.night_window()) * cfg.sleep_bonus_mult
    drain = cfg.energy_drain * c.traits.metabolism
    assert np.allclose(c.E[resting], (before + gain - drain)[resting])


def test_the_flag_expires_after_sixty_ticks():
    c = core()
    at_tick(c, 130)
    send(c, GO_SOCIAL)
    c.step(np.full(c.cfg.N, GO_SOCIAL, dtype=np.int64))
    assert (c._sleep_bonus_until > c.t).any(), "the flag is live immediately after company"
    for _ in range(c.cfg.sleep_bonus_ticks + 2):
        c.step(np.full(c.cfg.N, IDLE, dtype=np.int64))
    assert not (c._sleep_bonus_until > c.t).any(), "and dead once the window has passed"
    c.E[:] = 0.3
    send(c, GO_HOME)
    resting = c.active & (c.zone == HOME)
    assert resting.any()
    assert not c._sleep_bonus_stay[resting].any(), "so no bonus on this arrival"


def test_the_bonus_lasts_the_whole_stay_once_earned():
    """It is a night's sleep bought by the evening, not a rate that stops mid-sleep."""
    c = core()
    at_tick(c, 150)
    send(c, GO_SOCIAL)
    c.step(np.full(c.cfg.N, GO_SOCIAL, dtype=np.int64))
    c.E[:] = 0.1
    send(c, GO_HOME)
    resting = c.active & (c.zone == HOME)
    assert c._sleep_bonus_stay[resting].all()
    for _ in range(c.cfg.sleep_bonus_ticks + 20):            # well past the flag's expiry
        c.step(np.full(c.cfg.N, GO_HOME, dtype=np.int64))
    still = c.active & (c.zone == HOME)
    assert not (c._sleep_bonus_until > c.t).any(), "the flag itself is long gone"
    assert c._sleep_bonus_stay[still].all(), "but the stay keeps its bonus"


def test_leaving_home_ends_the_bonus():
    c = core()
    at_tick(c, 150)
    send(c, GO_SOCIAL)
    c.step(np.full(c.cfg.N, GO_SOCIAL, dtype=np.int64))
    c.E[:] = 0.3
    send(c, GO_HOME)
    assert c._sleep_bonus_stay[c.active & (c.zone == HOME)].all()
    send(c, GO_MARKET)
    assert not c._sleep_bonus_stay.any(), "the stay is over, so the bonus is spent"


# ---------------------------------------------------------------- coins
def test_starting_coins_are_the_v3_purse():
    """Economy v3 halved the starting purse to 10; food and drink are now the only sink."""
    assert HamletConfig().init_wealth == 10.0
    assert np.all(core().W == 10.0)


def test_no_wage_over_the_unpaid_hour():
    c = core()
    at_tick(c, 100)
    send(c, GO_FARM)
    working = c.active & (c.zone == FARM)
    assert working.any()
    # Productivity is min(1, E/floor) * min(1, F/floor), and an agent that has idled to midday is
    # flat out of energy. Top both up so the test measures the wage rule and not exhaustion.
    c.E[:], c.fast[:], c.slow[:] = 1.0, 0.5, 0.5
    c.step(np.full(c.cfg.N, GO_FARM, dtype=np.int64))
    assert np.all(c.coins_earned[working] > 0.0), "a normal hour pays"
    at_tick(c, 121)
    c.E[:], c.fast[:], c.slow[:] = 1.0, 0.5, 0.5
    c.step(np.full(c.cfg.N, GO_FARM, dtype=np.int64))
    still = c.active & (c.zone == FARM)
    assert still.any(), "the seat is kept"
    assert np.all(c.coins_earned[still] == 0.0), "but the hour pays nothing"


# ---------------------------------------------------------------- restlessness at home
def settle_at_home(c: HamletCore, energy: float = 1.0) -> None:
    """Put every agent at HOME, awake, with the given energy and a clean idle counter."""
    send(c, GO_HOME)
    c.E[:] = energy
    c._rest_idle_ticks[:] = 0


def test_no_restlessness_cost_while_tired():
    """Below rest_idle_energy the agent is resting, not idling, however long it stays."""
    c = core()
    at_tick(c, 100)
    settle_at_home(c, energy=0.5)
    for _ in range(c.cfg.rest_idle_after + 40):
        c.E[:] = 0.5                                  # held below the bar
        c.step(np.full(c.cfg.N, GO_HOME, dtype=np.int64))
        assert np.all(c.rest_idle_cost == 0.0)
    assert np.all(c._rest_idle_ticks == 0), "and the counter never starts"


def test_no_restlessness_cost_before_the_free_ticks_are_used():
    c = core()
    at_tick(c, 100)
    settle_at_home(c)
    cfg = c.cfg
    for k in range(cfg.rest_idle_after):
        c.E[:] = 1.0
        c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
        assert np.all(c.rest_idle_cost == 0.0), f"still free at tick {k + 1}"
    assert np.all(c._rest_idle_ticks == cfg.rest_idle_after)
    c.E[:] = 1.0
    c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert np.allclose(c.rest_idle_cost, cfg.rest_idle_cost), "the next tick is the first charged"


def test_no_restlessness_cost_at_night_at_any_energy():
    """Sleeping through the night is never charged, whatever the energy level."""
    c = core()
    at_tick(c, 221)                                    # inside the world's night
    settle_at_home(c, energy=1.0)
    for _ in range(c.cfg.rest_idle_after + 30):
        c.E[:] = 1.0
        c.step(np.full(c.cfg.N, GO_HOME, dtype=np.int64))
        if c.cfg.is_night(c.t_day):
            assert np.all(c.rest_idle_cost == 0.0)
            assert np.all(c._rest_idle_ticks == 0)


def test_the_cost_accrues_and_then_caps():
    c = core()
    at_tick(c, 100)
    settle_at_home(c)
    cfg = c.cfg
    seen = []
    for _ in range(cfg.rest_idle_after + 20):
        c.E[:] = 1.0
        c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
        seen.append(float(c.rest_idle_cost[0]))
    charged = [v for v in seen if v > 0.0]
    assert charged == sorted(charged), "the charge only grows while the agent stays"
    assert max(seen) == pytest.approx(cfg.rest_idle_cap), "and stops at the cap"
    assert all(v <= cfg.rest_idle_cap + 1e-12 for v in seen)
    # the cap is reached exactly where the arithmetic says it should be
    steps_to_cap = int(round(cfg.rest_idle_cap / cfg.rest_idle_cost))
    assert seen[cfg.rest_idle_after + steps_to_cap - 1] == pytest.approx(cfg.rest_idle_cap)


def test_the_counter_resets_when_the_agent_leaves_home():
    c = core()
    at_tick(c, 100)
    settle_at_home(c)
    cfg = c.cfg
    for _ in range(cfg.rest_idle_after + 6):
        c.E[:] = 1.0
        c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert np.all(c.rest_idle_cost > 0.0)
    send(c, GO_MARKET)                                 # leaving clears it
    assert np.all(c._rest_idle_ticks == 0)
    assert np.all(c.rest_idle_cost == 0.0)


def test_the_counter_resets_when_energy_drops():
    c = core()
    at_tick(c, 100)
    settle_at_home(c)
    cfg = c.cfg
    for _ in range(cfg.rest_idle_after + 6):
        c.E[:] = 1.0
        c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert np.all(c.rest_idle_cost > 0.0)
    c.E[:] = 0.5                                       # no longer rested
    c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert np.all(c._rest_idle_ticks == 0)
    assert np.all(c.rest_idle_cost == 0.0)


def test_the_counter_resets_at_ten_at_night():
    c = core()
    # Start early enough that the free ticks are used up and a charge is running by 22:00;
    # settling costs travel_ticks + 1 ticks before the counter can even begin.
    at_tick(c, 160)
    settle_at_home(c)
    cfg = c.cfg
    while c.t_day < 219:
        c.E[:] = 1.0
        c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert np.all(c.rest_idle_cost > 0.0), "charged right up to 22:00"
    # The rule reads the tick being processed, so the reset lands on the first step whose
    # tick-of-day is 220. Step until that tick is the one just applied.
    while not cfg.is_night(c._step_t_day):
        c.E[:] = 1.0
        c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert c._step_t_day == cfg.night[0], "the first night tick is 22:00"
    assert np.all(c._rest_idle_ticks == 0), "the counter resets at 22:00"
    assert np.all(c.rest_idle_cost == 0.0)


def test_the_charge_is_in_the_reward_and_not_in_the_drive():
    """D is what dashboards report; the reward carries the charge. The two must not be confused."""
    c = core()
    at_tick(c, 100)
    settle_at_home(c)
    cfg = c.cfg
    for _ in range(cfg.rest_idle_after + 6):
        c.E[:] = 1.0
        _, rew, _ = c.step(np.full(cfg.N, GO_HOME, dtype=np.int64))
    assert np.all(c.rest_idle_cost > 0.0)
    expected = -(c.D) - c.effort_penalty - c.rest_idle_cost
    assert np.allclose(rew, expected.astype(np.float32), atol=1e-6)
    assert np.allclose(c.D, c._drive()), "D carries no trace of the charge"


# ---------------------------------------------------------------- the tank mapping, asserted
def test_a_market_meal_fills_the_slow_tank_only():
    """Locked spec: the plain staple bought at the stall lands in the slow tank, and nowhere else."""
    c = core(**TWO_TANK)
    at_tick(c, 100)
    send(c, GO_MARKET)
    cfg = c.cfg
    # Each tank caps at cfg.tank_cap, so start low enough for a whole meal to fit untruncated.
    c.fast[:], c.slow[:], c.W[:] = 0.05, 0.05, 50.0
    fast0, slow0 = c.fast.copy(), c.slow.copy()
    c.step(np.full(cfg.N, GO_MARKET, dtype=np.int64))
    bought = c.meal_type == 2      # "slow": the tank a market meal fills
    assert bought.any(), "an affordable market meal is taken"
    ap = c.traits.appetite
    # A staple fills the slow half outright, so the tank lands on its cap less this tick's drain.
    assert np.allclose(c.slow[bought], (cfg.tank_cap - cfg.slow_tank_drain * ap)[bought]), \
        "the slow tank is filled to its cap"
    assert np.allclose(c.fast[bought], (fast0 - cfg.fast_tank_drain * ap)[bought]), \
        "the fast tank only drains; a market meal never touches it"


def test_a_food_street_meal_fills_the_fast_tank_only():
    """Locked spec: eating on the spot tops up the fast tank, and nowhere else."""
    c = core(**TWO_TANK)
    at_tick(c, 150)
    send(c, GO_CANTEEN)
    cfg = c.cfg
    seated = c.active & (c.zone == CANTEEN)
    assert seated.any()
    c.fast[:], c.slow[:], c.W[:] = 0.05, 0.05, 50.0
    fast0, slow0 = c.fast.copy(), c.slow.copy()
    c.step(np.full(cfg.N, GO_CANTEEN, dtype=np.int64))
    ap = c.traits.appetite
    assert np.allclose(c.fast[seated], (fast0 + cfg.fast_tank_fill - cfg.fast_tank_drain * ap)[seated]), \
        "the fast tank takes the meal"
    assert np.allclose(c.slow[seated], (slow0 - cfg.slow_tank_drain * ap)[seated]), \
        "the slow tank only drains; a Food Street meal never touches it"


# ---------------------------------------------------------------- farm feeds, office pays
FARM_FEEDS = dict(farm_feeds=True)


def test_farm_feeds_is_on_by_default_in_v3():
    """Economy v3 makes the two jobs different: the farm feeds and pays a quarter."""
    cfg = HamletConfig()
    assert cfg.farm_feeds is True
    assert cfg.office_wage == cfg.wage
    assert cfg.farm_wage == pytest.approx(cfg.wage * cfg.farm_wage_share)


def test_the_farm_pays_half_of_what_the_office_pays():
    cfg = HamletConfig(**FARM_FEEDS)
    assert cfg.office_wage == cfg.wage, "the office is unchanged from v1"
    assert cfg.farm_wage == pytest.approx(cfg.wage * cfg.farm_wage_share)
    assert cfg.farm_wage == pytest.approx(cfg.office_wage * cfg.farm_wage_share)


def test_a_productive_farm_tick_feeds_the_worker():
    c = core(**FARM_FEEDS)
    at_tick(c, 100)
    send(c, GO_FARM)
    cfg = c.cfg
    working = c.active & (c.zone == FARM)
    assert working.any()
    c.E[:], c.slow[:] = 1.0, 0.3          # productivity pinned high, room in the tank
    before = c.slow.copy()
    c.step(np.full(cfg.N, GO_FARM, dtype=np.int64))
    assert np.all(c.farm_fed[working] > 0.0), "the farm feeds while it is worked"
    assert np.all(c.slow[working] > before[working] - cfg.satiety_drain_v1), "so satiety rises"
    assert np.all(c.farm_fed[~working] == 0.0), "and only the workers are fed"


def test_the_office_never_feeds():
    c = core(**FARM_FEEDS)
    at_tick(c, 100)
    send(c, GO_OFFICE)
    working = c.active & (c.zone == OFFICE)
    assert working.any()
    c.E[:], c.slow[:] = 1.0, 0.3
    c.step(np.full(c.cfg.N, GO_OFFICE, dtype=np.int64))
    assert np.all(c.farm_fed == 0.0), "the office pays, it does not feed"


def test_the_farm_costs_more_energy_than_the_office():
    cfg = HamletConfig(**FARM_FEEDS)
    drops = {}
    for action, zone in ((GO_FARM, FARM), (GO_OFFICE, OFFICE)):
        c = core(**FARM_FEEDS)
        at_tick(c, 100)
        send(c, action)
        working = c.active & (c.zone == zone)
        assert working.any()
        c.E[:] = 1.0
        c.step(np.full(cfg.N, action, dtype=np.int64))
        drops[zone] = float(1.0 - c.E[working].mean())
    assert drops[FARM] > drops[OFFICE], "the farm is the harder day's work"
    extra = cfg.energy_work_drain * (cfg.farm_energy_mult - 1.0)
    assert drops[FARM] - drops[OFFICE] == pytest.approx(extra, rel=0.2)


def test_the_farm_feeds_nothing_when_the_flag_is_off():
    c = core(farm_feeds=False)
    at_tick(c, 100)
    send(c, GO_FARM)
    assert (c.active & (c.zone == FARM)).any()
    c.E[:], c.slow[:] = 1.0, 0.3
    c.step(np.full(c.cfg.N, GO_FARM, dtype=np.int64))
    assert np.all(c.farm_fed == 0.0)


# ---------------------------------------------------------------- economy v3: start trigger
def test_a_bar_visit_is_one_drink_and_lasts_until_the_set_point():
    """Rule 3 with the start trigger: enter below 0.5, one charge, stay until 0.9."""
    c = core()
    at_tick(c, 130)
    cfg = c.cfg
    # The start trigger is GREEDY_SERVE_TRIGGER lowered by the laziness trait:
    # 0.5 - greedy_threshold_drop * laziness. The neutral population has laziness 0, so here the
    # trigger is 0.5 exactly; it is a lazy population that acts as late as 0.2.
    assert (c.traits.laziness == 0.0).all()
    assert GREEDY_SERVE_TRIGGER - cfg.greedy_threshold_drop * 0.0 == pytest.approx(0.5)
    # Energy and satiety are held well above the trigger so that social is the only eligible need.
    # Left where the walk to tick 130 put them, energy is the larger deficit, the agent rests, and
    # the commitment holds it at home until energy reaches GREEDY_COMMIT_UNTIL: no bar visit, and
    # nothing about the bar rule measured.
    c.E[:] = 0.95
    c.F[:] = 0.95
    c.C[:] = 0.20
    c.W[:] = 50.0
    pol = GreedyClockPolicy()
    charges, at_bar = 0, 0
    for _ in range(60):                             # one visit is about 25 ticks
        a, _ = pol.act(c._observe(), c)
        c.step(a)
        charges += int(c.coins_spent_drink[0] > 0)
        at_bar += int(c.zone[0] == SOCIAL)
        if c.C[0] >= 0.9:
            break
    assert at_bar > 0, "the agent goes to the bar when social is below the trigger"
    assert charges == 1, f"one visit, one drink charge, got {charges}"
    assert c.C[0] > 0.45, "and social actually rises while it is there"


def test_a_fed_solvent_agent_in_job_hours_takes_the_office():
    """Rule 4: the farm is for the hungry, the office for everyone else."""
    c = core()
    at_tick(c, 100)
    cfg = c.cfg
    assert cfg.job_is_open(c.t_day)
    c.E[:], c.C[:] = 1.0, 1.0
    c.slow[:], c.F[:] = 0.95, 0.95                  # well fed: above the start trigger
    c.W[:] = 1.0                                    # below daily_budget x margin, so rule 4 fires
    a, _ = GreedyClockPolicy().act(c._observe(), c)
    assert np.all(a == GO_OFFICE), f"a fed agent works the office, got {a.tolist()}"


def test_a_hungry_agent_in_job_hours_takes_the_farm():
    """The farm feeds while it pays, so a hungry agent earns and eats at once."""
    c = core()
    at_tick(c, 100)
    cfg = c.cfg
    assert cfg.job_is_open(c.t_day)
    c.E[:], c.C[:] = 1.0, 1.0
    c.slow[:], c.F[:] = 0.2, 0.2                    # below the start trigger and below the feed cap
    c.W[:] = 1.0
    a, _ = GreedyClockPolicy().act(c._observe(), c)
    assert np.all(a == GO_FARM), f"a hungry agent works the farm, got {a.tolist()}"


def test_the_farm_stops_feeding_at_its_cap_but_keeps_paying():
    c = core()
    at_tick(c, 100)
    send(c, GO_FARM)
    cfg = c.cfg
    working = c.active & (c.zone == FARM)
    assert working.any()
    c.E[:], c.slow[:] = 1.0, cfg.farm_feed_cap + 0.05    # already past the cap
    c.step(np.full(cfg.N, GO_FARM, dtype=np.int64))
    assert np.all(c.farm_fed[working] == 0.0), "past the cap the farm stops feeding"
    assert np.all(c.coins_earned[working] > 0.0), "but it still pays"
