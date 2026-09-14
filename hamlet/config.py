"""Every constant of the Hamlet world lives here and nowhere else.

The one exception is the table of trait caps, ``hamlet/traits.py::CAPS``,
which belongs with the trait definitions; the trait-related world constants
(the generator salt, the trait block width, the effort and threshold
coefficients) are here.

The world has seven zones: the agents' own HOME tiles, two jobs (FARM and
OFFICE) that pay the same wage during the same hours and each hold
``ceil(N/2)`` workers, a CANTEEN, a SOCIAL square, a MARKET and TRANSIT for
every other tile. Seven actions pick a destination (one per zone that can be
requested) or IDLE.

The dataclass is printed into every run's metadata and reproduced as a table in
the paper. Changing a value after the analysis plan is tagged is a deviation and
must be recorded in the deviations section of ``docs/analysis_plan.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json
import math

Arm = Literal["A", "B", "C", "D", "E"]
Symmetry = Literal["S0", "S1"]
Shock = Optional[Literal["jobs_closed_d4", "energy_x2_d4"]]

# Zone ids (also the index of the one-hot "current zone" block in the observation)
HOME, FARM, OFFICE, CANTEEN, SOCIAL, MARKET, TRANSIT = 0, 1, 2, 3, 4, 5, 6
ZONE_NAMES = ("HOME", "FARM", "OFFICE", "CANTEEN", "SOCIAL", "MARKET", "TRANSIT")
# The two jobs. Both pay the same wage during the same hours and each has its
# own capacity, so choosing between them is never a time-of-day choice.
JOBS = (FARM, OFFICE)

# Action ids
GO_HOME, GO_FARM, GO_OFFICE, GO_CANTEEN, GO_SOCIAL, GO_MARKET, IDLE = 0, 1, 2, 3, 4, 5, 6
ACTION_NAMES = ("GO_HOME", "GO_FARM", "GO_OFFICE", "GO_CANTEEN", "GO_SOCIAL", "GO_MARKET", "IDLE")
N_ACTIONS = 7

# Zones that have a fixed rectangle (HOME tiles are per agent, see HamletConfig.home_x)
# Rectangles are inclusive (x0, y0, x1, y1) on a grid with origin top-left.
# FARM keeps the rectangle of the former single workplace; OFFICE sits at the
# bottom left, clear of the home columns, the canteen, the market and the square.
ZONE_RECTS = {
    FARM:    (14, 1, 18, 4),
    OFFICE:  (3, 18, 7, 19),
    MARKET:  (8, 1, 11, 3),
    CANTEEN: (8, 8, 11, 10),
    SOCIAL:  (13, 14, 18, 18),
}
# Order of the five public zones in the distance / occupancy / open-flag blocks
PUBLIC_ZONES = (FARM, OFFICE, CANTEEN, SOCIAL, MARKET)

# Home tiles: column HamletConfig.home_x, rows HOME_Y_FIRST + HOME_Y_STEP * k.
# When more agents are requested than one column holds, the overflow continues
# on a second column HOME_X_STEP tiles to the right with the same rows.
HOME_Y_FIRST, HOME_Y_STEP, HOME_X_STEP = 2, 2, 2

# Thresholds of the hand-written greedy baselines (GreedyStatePolicy, GreedyClockPolicy).
GREEDY_MARKET_HUNGER = 0.3    # satiety below which an agent that can afford it buys a market meal
GREEDY_NIGHT_REST = 0.9       # energy below which GreedyClock goes home during the night
GREEDY_COMMIT_UNTIL = 0.9
# Economy v3: a need is not served at all until it falls below this, and is then
# served to GREEDY_COMMIT_UNTIL before the scheduler will switch. Without a start trigger every
# level below 1.0 counted as a deficit, so the scheduler nibbled: it visited the bar fifteen times
# a day at a drink a visit, and a well-fed agent still counted as hungry, which sent it to the farm
# rather than the office. The laziness trait lowers it further, so a lazy agent acts later still.
GREEDY_SERVE_TRIGGER = 0.5     # a greedy agent keeps serving its chosen need until that level is reached
GREEDY_WORK_ABOVE = 0.5       # GREEDY-CLOCK-WORK (job-first) works unless its lowest need is above this level
# Night rule of the clock-aware scheduler. "hysteresis" is the definition used as the routine
# reference line: an agent with energy below GREEDY_NIGHT_REST goes home at night and stays
# there until the night ends. "literal" re-evaluates every tick (oscillates at the threshold);
# "hysteresis_wake" releases the agent once energy is full. Both kept for the calibration report.
GREEDY_NIGHT_RULE = "hysteresis"
GREEDY_NIGHT_RULES = ("literal", "hysteresis", "hysteresis_wake")

# Clock: hours in a day, used to turn the chronotype offset (hours) into ticks.
HOURS_PER_DAY = 24

# Traits. The integer salt of the per-agent trait generator
# ``default_rng([run_seed, TRAITS_SALT, agent_id])`` (see hamlet/traits.py); it keeps
# the trait draws apart from every world stream, which use the bare run seed.
TRAITS_SALT = 20260822
# Width of the trait block in the observation: the five scalar traits (appetite,
# metabolism, chronotype, laziness, learning_rate) plus one aptitude entry per job.
N_SCALAR_TRAITS = 5
TRAIT_VECTOR_LEN = N_SCALAR_TRAITS + len(JOBS)


@dataclass
class HamletConfig:
    # ---- population and world -------------------------------------------------
    # Capacities follow the registered formulas
    # and move with it: canteen ceil(N/4) = 3, each job ceil(N/2) = 5, market and social unlimited.
    n_agents: int = 9
    grid: int = 20                      # grid is grid x grid, open, Manhattan travel
    home_x: int = 1                     # home tiles at (home_x, 2 + 2*k), permuted per episode
    # Capacities (None = unlimited). Evaluated lazily from n_agents.
    cap_job: Optional[int] = None       # default ceil(N/2), for EACH job (FARM and OFFICE)
    cap_canteen: Optional[int] = None   # default ceil(N/4)
    cap_social: Optional[int] = None    # default unlimited
    cap_market: Optional[int] = None    # default unlimited

    # ---- time -----------------------------------------------------------------
    ticks_per_day: int = 240            # one tick = 6 game minutes
    n_days: int = 6                     # episode length in days; truncation only, never termination
    burn_in_days: int = 1               # days excluded from every metric
    # World v2. Every hour below is a locked part of the v2 spec, not a
    # calibrated value: they are the opening times the village runs on.
    job_open: tuple[int, int] = (70, 190)      # both jobs open 07:00-19:00
    # Lunch: the jobs stay open and nobody is evicted, but no wage is paid 12:00-13:00. An agent
    # may sit at its desk through lunch; it simply earns nothing for that hour.
    job_unpaid: tuple[int, int] = (120, 130)
    # Opening hours apply to the jobs only. v2-lite put hours on four
    # zones at once and the reference scheduler spent 48% of its ticks walking between places that
    # were shut when it arrived, against 26% in the v1 world. The jobs keep their hours because
    # that is what a routine entrains to; everything else is open all day, so a need can always be
    # served somewhere and travel is a cost rather than a gamble.
    # Economy v3. The market and Food Street keep hours; the bar does not,
    # because company is the need the village was worst at serving and shutting it made that worse.
    market_open: tuple[int, int] = (70, 190)   # the stall trades 07:00-19:00
    canteen_open: tuple[int, int] = (110, 230) # Food Street serves 11:00-23:00
    bar_open: tuple[int, int] = (0, 240)       # always open, no eviction
    night: tuple[int, int] = (220, 60)         # night = t_day >= 220 or t_day < 60 (22:00-06:00)
    night_rest_bonus: float = 0.5       # rest restores (1 + bonus) x faster at night
    # Travel: every journey between two different zones costs this many TRANSIT ticks, whichever
    # pair of zones it joins. It replaces Manhattan movement over the grid.
    # Origin: the mean length of a completed TRANSIT run under GREEDY-CLOCK on the 32 evaluation
    # episodes of runs/A-S1-N8/, 9.446 ticks over 8,604 runs, rounded to the nearest integer.
    travel_ticks: int = 9

    # ---- internal-state dynamics (per tick, levels clipped to [0, 1]) ---------
    energy_drain: float = 0.005         # awake, every tick
    energy_work_drain: float = 0.005    # extra while actively working at either job
    energy_rest_gain: float = 0.0175    # while actively resting at own home (gross; net = gain - drain)
    # ---- hunger: two tanks (world v2) ----------------------------------------------------
    # Satiety is no longer one level. It is the sum of a fast tank and a slow tank, clipped at 1:
    #
    #     F = min(1, fast + slow)
    #
    # The two drain at different rates, so what an agent ate and when both matter. A market meal
    # is instant and fills the fast tank; it is cheap and does not last. A Food Street meal fills
    # the slow tank while the agent sits there, costing coins every tick; it is dearer and lasts.
    # Both debit coins, and a meal an agent cannot afford is refused outright, with no gain, no
    # debit and meal_type "none".
    #
    # The six numbers below are starting values, calibrated against the scripted scheduler's
    # competence gate. They are set to reproduce v1's
    # behaviour at the start: fast + slow drain to 0.006 against v1's single 0.003 (the tanks
    # halve, so the sum drains at a comparable rate per unit of satiety), and the two meal gains
    # keep v1's magnitudes.
    # Calibrated 0.005 -> 0.0015 against the design constraint that
    # one market meal plus one Food Street sitting keeps satiety above 0.3 for a whole day. At
    # 0.005 a full fast tank emptied in 200 ticks, so a 30-tick sitting was gone before the
    # evening and satiety rested on the slow tank alone.
    fast_tank_drain: float = 0.003
    slow_tank_drain: float = 0.001      # v1 lineage: not a free parameter, stays put
    # A staple fills the slow half outright rather than adding a fixed amount, so this is no longer
    # a gain: see market_price above.
    # Calibrated 0.024 -> 0.035, per active Food Street tick. A
    # 30-tick sitting now fills 1.05 rather than 0.72, which with the drain above carries the fast
    # tank past midnight.
    fast_tank_fill: float = 0.035       # per seated tick, until the fast half reaches its cap
    # Each tank holds at most half of satiety. F = slow + fast, so
    # neither counter can feed an agent on its own: the staple bought at the market fills the slow
    # half and a Food Street sitting fills the fast half, and an agent that wants satiety above 0.5
    # has to use both. A fill that would overflow its tank is truncated at the counter and the meal
    # still costs full price, which is the trade-off for arriving already full.
    tank_cap: float = 0.5

    # ---- v2-lite -------------------------------------------------------------------------
    # The two-tank economy is off by default. v2-lite is world v2 without it: the bar with its
    # staff, floor and slope; the sleep bonus; the restlessness charge; and every opening hour,
    # all on top of v1's single satiety level and its free canteen.
    #
    # The two-tank economy was built to the spec and could not be calibrated. With each half
    # capped at 0.5 the cheap staple was permanently affordable and permanently useful, so the
    # scheduler churned staples, never saved the 8 coins a sitting costs, and left the fast half
    # empty; the competence gate never rose above 0.23 at any wage in [0.3, 20], at any price
    # ratio tried, or with the trigger at 0.5 or 0.40. It is kept behind this flag so the work is
    # not lost and the design can be revisited with a different satiety shape.
    two_tank_economy: bool = False

    # ---- farm feeds, office pays ---------------------------------------------------------
    # Off by default. With it on the two jobs stop being interchangeable: the farm feeds you a
    # little while you work it and costs you more energy, and it pays half what the office does.
    # The office is unchanged from v1. The point is to give the two jobs different characters so
    # that choosing between them carries information, rather than being a coin toss the capacity
    # rule settles. No money sink is added today, so coins simply accumulate; that is noted for v3.
    # Economy v3 default. The farm feeds its workers while they earn, at eating speed, for a
    # modest extra energy cost and half the office wage. That makes the two jobs genuinely
    # different: the farm is the poor man's lunch, the office is the money.
    farm_feeds: bool = True
    farm_feed_rate: float = 0.024       # per productive farm tick, the same speed as eating
    # The farm feeds only so far. Past this the work continues and still pays, but it stops
    # filling: a farm day leaves the agent fed enough to keep going and short of the 0.9 set-point,
    # so it still has to buy a proper meal. Without the cap the farm made both money and food
    # unnecessary, and the office was never chosen at any wage.
    farm_feed_cap: float = 0.6
    farm_energy_mult: float = 1.25      # the farm's work drain, times the ordinary one
    farm_wage_share: float = 0.25       # the farm pays this share of the office wage

    # There is no free food in v3. The CANTEEN zone is Food Street's counter and charges for a
    # seat. free_canteen restores the v2-lite behaviour for comparison.
    free_canteen: bool = False
    # A seat at Food Street is charged once, on the tick it is taken, and fills satiety outright:
    # the expensive, instant meal. Its capacity follows the registered formula, three seats at N=9.
    food_street_price: float = 8.0
    # A market sitting is charged once and fills at eating speed while the agent stays: the cheap,
    # slow meal.
    market_price: float = 3.0
    # A drink is charged once when the agent becomes active at the bar, and is refused if the purse
    # is short. Company itself is unpriced; the drink is the cover charge.
    drink_price: float = 3.0
    # v1 lineage, used when the flag is off: the canteen is free and fills satiety while seated,
    # and a market meal is an instant top-up that costs market_price_v1.
    satiety_drain_v1: float = 0.003
    canteen_gain_v1: float = 0.024      # eating speed: the rate a market sitting fills at
    # Calibrated against the scripted scheduler's competence gate, 12.0 -> 2.0, coins per market staple. Once each tank
    # capped at half of satiety the staple became necessary rather than optional, and at 12 coins
    # against a wage of 4 the scheduler could never save for one: Food Street costs 0.5 a tick and
    # drained the purse faster than work filled it, so it ate 16 sittings a day and bought no
    # staple at all, leaving the slow tank at 0.048 and the gate at 0.858. Raising the wage to 12
    # also clears the gate, but only because it makes one tick of work buy exactly one staple; that
    # knife-edge breaks as soon as productivity drops below 1. A wage-to-price ratio of 2 clears it
    # on ten seeds with a minimum of 1.000 and no such dependence.
    # A staple costs this and fills the slow half to its cap in one purchase.
    market_price: float = 3.0
    # Food Street charges once per sitting, not per tick. The charge
    # falls on the tick the seat is taken; the sitting then fills the fast half to its cap over the
    # following ticks at no further cost. Leaving and coming back is a new sitting and a new charge,
    # so a scheduler that dithers pays twice.
    food_street_price: float = 8.0
    # The budget rule works until the purse covers a day's eating times this margin.
    # At exactly 1.0 the agent stops the moment it can afford today and never builds a buffer, so
    # one unproductive tick puts it short again and a sitting becomes unaffordable.
    budget_margin: float = 1.5
    init_fast_low: float = 0.25         # starting value; fast tank at reset
    init_fast_high: float = 0.50
    init_slow_low: float = 0.25         # starting value; slow tank at reset
    init_slow_high: float = 0.50
    social_drain: float = 0.002
    social_gain: float = 0.022          # per active bar tick, before the company multiplier below

    # ---- the bar (world v2) --------------------------------------------------------------
    # SOCIAL is the bar. The code identifier does not change, following the rule in
    # scripts/analysis_constants.py that identifiers are fixed and only display names move;
    # the human name is set there. The bar keeps a fixed staff presence: staff have no needs,
    # take no decisions and never appear as a row in the log, so they are not agents. Their
    # only effect is that an agent drinking alone is not drinking in an empty room.
    #
    # Company gain = social_gain * (bar_solo_share + bar_company_step * min(others, cap)),
    # where `others` counts villagers only, never the staff. So a solo visit earns half, one
    # companion three quarters, and two or more the full rate. This replaces the v1 rule, where
    # a solo visit earned social_solo_fraction (0.0 in every registered run) and company was all
    # or nothing.
    bar_solo_share: float = 0.5         # locked by the v2 spec
    bar_company_step: float = 0.25      # locked by the v2 spec
    bar_company_cap: int = 2            # locked by the v2 spec

    # ---- restlessness at home (world v2) -------------------------------------------------
    # A small per-tick cost for sitting at home once there is nothing left to gain from it. An
    # agent that has restored its energy and stays put anyway is idling, and without a cost the
    # cheapest policy in a homeostatic world is to go home early and never leave. The cost is in
    # drive units and is added to the reward, not to D: dashboards report the drive the agent
    # actually carries, and the reward carries the nudge. Logged as its own column so the reward
    # stream stays decomposable.
    #
    # Sleeping through the night is never charged, whatever the energy level, so the cost cannot
    # punish an agent for keeping ordinary hours.
    rest_idle_energy: float = 0.9       # "rested" for this purpose; matches GREEDY_NIGHT_REST
    rest_idle_after: int = 30           # starting value; free ticks first
    rest_idle_cost: float = 0.05        # starting value; per further tick
    rest_idle_cap: float = 0.30         # starting value; never exceeds this

    # ---- the sleep bonus (world v2) ------------------------------------------------------
    # An agent that earns any company gain carries a flag for sleep_bonus_ticks ticks. If the
    # flag is live at the moment it arrives home, rest restores sleep_bonus_mult times faster
    # for that whole stay, even after the flag itself expires. An evening at the bar buys a
    # better night's sleep.
    sleep_bonus_ticks: int = 60         # locked by the v2 spec
    sleep_bonus_mult: float = 1.25      # locked by the v2 spec
    # Exploratory only (arm E2), outside every confirmatory family. An agent active at
    # SOCIAL with nobody else present receives this fraction of social_gain. The default 0.0 is the
    # registered world exactly: co-presence is required, so every confirmatory run and the pinned
    # trajectory fixture are unchanged. Raising it turns the social need from a coordination problem
    # into one an agent can partly solve alone, which is the question E2 asks.
    social_solo_fraction: float = 0.0
    # Calibrated against the scripted scheduler's competence gate, 1.0 -> 4.0, coins per active job tick, times
    # productivity, the same at both jobs. World v2 made both meals cost coins, and at 1.0 the
    # scheduler could not earn one market meal a day: it ended every day with a purse of about 0.3,
    # bought no Food Street meal at all in five days, and spent 41% of its ticks in transit between
    # counters it could not pay at. The competence gate read 0.16 to 0.23 against the 0.90 rule.
    # The knee is sharp: 3.0 gives a minimum of 0.832 over six seeds, 3.5 gives 0.965 and 4.0
    # gives 0.986, so 4.0 sits clear of the cliff rather than on it.
    # Economy v3: coins per productive office tick. It is small because
    # food and drink are now the only sink and the day is 240 ticks long.
    # Economy v3, chosen from a sweep of office_wage {1.2, 1.6, 2.0, 2.4} against
    # budget_margin {1.5, 2.5, 3.5}, seeds 0-9 a cell, gate as the minimum over seeds.
    # That first sweep ran before the rule 4 fix, when a solvent, well-fed
    # agent fell through to job_choice and picked its job by aptitude and tie-break rather than
    # taking the office, so it is re-measured here against the scheduler as it stands.
    # Margin 1.5 clears the 0.90 gate at wages 1.6 (0.928), 2.0 (0.933) and 2.4 (0.934), and the
    # fix lifted every cell: the same three read 0.902, 0.922 and 0.927 before it. 2.0 is kept.
    # The three clearing cells sit within 0.006 of each other, so the re-measurement does not
    # displace the value the pilot ran on, and the purse drift over an episode grows with the wage
    # (1.8, 5.0, 10.7 coins), which is a reason not to move up rather than a reason to.
    # Two readings from the first sweep do not survive the fix. Farm ticks are flat near 17.3 a day
    # at every wage, not traded against energy, so "a better-paid agent works the farm less" was an
    # artefact of the job_choice fall-through. And the office is still barely used at any wage:
    # 1.6 ticks an agent-day at the chosen cell, 1.7 to 3.2 across the rest. Rule 4 sends a hungry
    # agent to the farm because the farm feeds, and the budget rule only fires while the purse is
    # short, so an agent that is neither hungry nor poor has no reason to go. That is an open
    # point, recorded rather than closed: the office target is not met at any cell of this sweep.
    wage: float = 2.0
    init_wealth: float = 10.0           # starting purse, economy v3
    wealth_obs_scale: float = 60.0      # observation uses min(W / scale, 1)
    mood_tau: float = 0.95              # M <- tau*M + (1-tau)*(1 - D/3)
    init_mood: float = 0.7
    mood_coupling: bool = False         # if True productivity *= (0.5 + 0.5*M)
    productivity_floor: float = 0.2     # P = min(1, E/floor) * min(1, F/floor)
    init_state_low: float = 0.5         # E, F, C ~ U(low, high) at reset
    init_state_high: float = 1.0

    # ---- reward ---------------------------------------------------------------
    w_energy: float = 1.0
    w_satiety: float = 1.0
    w_social: float = 1.0               # set to 0 by social_on=False
    reward_form: Literal["level", "difference"] = "level"   # -D_t  or  k*(D_t - D_{t+1})
    difference_scale: float = 10.0      # k for the difference form (sanity-check arm only)
    income_scale: float = 1.0           # arms C/D: r = income_scale * coins earned this tick

    # ---- experimental condition -----------------------------------------------
    arm: Arm = "A"                      # A: states seen, -D | B: hidden, -D | C: seen, income | D: hidden, income | E: proxies, -D
    symmetry: Symmetry = "S1"           # S0: agent-ID block zero-filled | S1: one-hot agent ID
    clock_visible: bool = True          # False = "A-noclock": sin/cos zero-filled, opening hours kept
    social_on: bool = True              # False: w_social = 0 and C zero-filled in obs; SOCIAL zone kept
    gossip: bool = True                 # measurement-only information bit, zero training cost
    shock: Shock = None                 # evaluation-time perturbation
    shock_day: int = 4                  # day index (0-based) on which a shock applies
    shock_energy_mult: float = 2.0

    # ---- traits (per-agent, read-only after construction; see hamlet/traits.py) -------
    population: str = "neutral"         # name in traits.POPULATIONS; "neutral" reproduces the trait-free world
    obs_include_traits: bool = True     # False zero-fills the trait block of the observation (never drops it)
    normalise_aptitude: bool = True     # aptitude vectors rescaled to geometric mean 1
    skill_max: float = 1.0              # asymptote of skill[job] (dimensionless; coins scale by 1 + skill)
    forgetting_rate: float = 0.0        # per-tick decay of skill at the jobs not worked this tick
    effort_penalty_coef: float = 0.02   # reward units per effort tick at laziness 1 (times wage in arms C/D)
    # Superseded by GREEDY_SERVE_TRIGGER in economy v3: the eligibility ceiling is the trigger,
    # not this field, which now gates nothing and is kept so an archived config still loads.
    greedy_base_threshold: float = 1.0  # was the greedy act threshold at laziness 0
    greedy_threshold_drop: float = 0.3  # start trigger = GREEDY_SERVE_TRIGGER - drop * laziness

    # ---- bookkeeping ----------------------------------------------------------
    proxy_scale: float = 480.0          # arm E: ticks since {rest, meal, social} / proxy_scale, clipped to 1

    # ---- derived --------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.n_agents

    @property
    def episode_ticks(self) -> int:
        return self.ticks_per_day * self.n_days

    @property
    def obs_dim(self) -> int:
        return OBS_FIXED + self.n_agents

    @property
    def ticks_per_hour(self) -> float:
        """Ticks in one game hour (``ticks_per_day / HOURS_PER_DAY``)."""
        return self.ticks_per_day / HOURS_PER_DAY

    def capacity(self, zone: int) -> Optional[int]:
        N = self.n_agents
        if zone in JOBS:
            return self.cap_job if self.cap_job is not None else math.ceil(N / 2)
        if zone == CANTEEN:
            return self.cap_canteen if self.cap_canteen is not None else math.ceil(N / 4)
        if zone == SOCIAL:
            return self.cap_social
        if zone == MARKET:
            return self.cap_market
        if zone == HOME:
            return 1
        return None

    def is_night(self, t_day: int) -> bool:
        a, b = self.night
        return t_day >= a or t_day < b

    @property
    def office_wage(self) -> float:
        """The office pays the registered wage, unchanged from v1."""
        return self.wage

    @property
    def farm_wage(self) -> float:
        """The farm pays a share of the office wage when it feeds; otherwise the same."""
        return self.wage * self.farm_wage_share if self.farm_feeds else self.wage

    @property
    def daily_budget(self) -> float:
        """Coins an ordinary day costs: one drink and one market meal.

        Derived, never set by hand. It is what the schedulers' budget rule aims at, so the food
        prices and the target purse cannot drift apart.
        """
        return self.drink_price + self.market_price

    def job_is_open(self, t_day: int) -> bool:
        """True while the jobs admit workers; FARM and OFFICE share the same hours.

        Open is not the same as paying. The jobs do not evict at close and do not pay over
        lunch; see ``job_pays``.
        """
        return self.job_open[0] <= t_day < self.job_open[1]

    def job_pays(self, t_day: int) -> bool:
        """True while a job tick earns a wage: open, and not the unpaid lunch hour."""
        return self.job_is_open(t_day) and not (self.job_unpaid[0] <= t_day < self.job_unpaid[1])

    def market_is_open(self, t_day: int) -> bool:
        return self.market_open[0] <= t_day < self.market_open[1]

    def canteen_is_open(self, t_day: int) -> bool:
        """True while Food Street serves. It evicts whoever is still seated at close."""
        return self.canteen_open[0] <= t_day < self.canteen_open[1]

    def bar_is_open(self, t_day: int) -> bool:
        """True while the bar serves. It evicts whoever is still there at close."""
        return self.bar_open[0] <= t_day < self.bar_open[1]

    @property
    def states_visible(self) -> bool:
        return self.arm in ("A", "C")

    @property
    def homeostatic_reward(self) -> bool:
        return self.arm in ("A", "B", "E")

    @property
    def condition_name(self) -> str:
        parts = [self.arm, self.symmetry, f"N{self.n_agents}"]
        if not self.clock_visible:
            parts.append("noclock")
        if not self.social_on:
            parts.append("socialoff")
        if self.mood_coupling:
            parts.append("mood")
        if self.shock:
            parts.append(self.shock)
        if self.population != "neutral":
            parts.append(f"pop_{self.population}")
        return "-".join(parts)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def validate(self) -> None:
        assert 2 <= self.n_agents <= 2 * (self.grid // 2 - 2), "homes must fit on the left column"
        assert self.burn_in_days < self.n_days
        assert self.arm in ("A", "B", "C", "D", "E")
        assert self.symmetry in ("S0", "S1")
        assert self.reward_form in ("level", "difference")
        from hamlet.traits import POPULATIONS  # imported here: traits.py imports this module

        assert self.population in POPULATIONS, f"unknown population {self.population!r}; choose from {sorted(POPULATIONS)}"
        assert self.skill_max > 0 and self.forgetting_rate >= 0 and self.effort_penalty_coef >= 0
        assert self.greedy_threshold_drop >= 0
        for z, (x0, y0, x1, y1) in ZONE_RECTS.items():
            assert 0 <= x0 <= x1 < self.grid and 0 <= y0 <= y1 < self.grid, z
        rects = list(ZONE_RECTS.items())
        for i, (za, a) in enumerate(rects):
            for zb, b in rects[i + 1:]:
                assert not _rects_overlap(a, b), f"zones {za} and {zb} overlap"
        for x, y in self.home_tiles():
            for z, r in ZONE_RECTS.items():
                assert not _inside(x, y, r), f"home tile ({x}, {y}) lies inside zone {z}"

    def home_tiles(self) -> list[tuple[int, int]]:
        """Home tiles in agent-index order before the per-episode permutation.

        Column ``home_x`` holds rows ``HOME_Y_FIRST + HOME_Y_STEP * k``; when
        ``n_agents`` exceeds what one column holds the overflow continues
        ``HOME_X_STEP`` tiles to the right with the same rows.
        """
        rows = (self.grid - 1 - HOME_Y_FIRST) // HOME_Y_STEP + 1
        return [
            (self.home_x + HOME_X_STEP * (k // rows), HOME_Y_FIRST + HOME_Y_STEP * (k % rows))
            for k in range(self.n_agents)
        ]


def _inside(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ---- observation layout (fixed part; agent-ID one-hot of length N follows) -------
# index : meaning
#  0- 2 : E, F, C                (zero-filled in arms B/D; C zero-filled if social_on=False;
#                                 replaced by elapsed-time proxies in arm E)
#  3- 4 : fast and slow hunger tank   (the same visibility rule as E, F, C; both carry the meal
#                                 proxy in arm E)
#  5    : min(W / wealth_obs_scale, 1)
#  6- 7 : (1 + sin) / 2, (1 + cos) / 2 of 2*pi*t_day/ticks_per_day   (zero-filled if clock_visible=False)
#  8- 9 : x, y                   (always zero: position left the simulation with the fixed travel rule)
# 10-14 : distance to FARM, OFFICE, CANTEEN, SOCIAL, MARKET   (always zero, as above)
# 15-19 : occupancy fraction of those five (active / capacity, or active / N if unlimited)
# 20-24 : open flag of those five (FARM and OFFICE share job_open, MARKET follows market_open,
#         CANTEEN and SOCIAL are always 1)
# 25    : distance to own home   (always zero, as above)
# 26-32 : one-hot current zone {HOME, FARM, OFFICE, CANTEEN, SOCIAL, MARKET, TRANSIT}
# 33-39 : one-hot previous action (seven actions; all zero on the first tick)
# 40    : active bit (activity granted last tick)
# 41-47 : traits, each mapped linearly by its cap into [0, 1]: appetite, metabolism,
#         chronotype, laziness, aptitude at FARM, aptitude at OFFICE, learning_rate
#         (all zero if obs_include_traits=False; the neutral population writes its constants)
# 48..  : one-hot agent ID (length N; all zero under S0)
_N_PUBLIC = len(PUBLIC_ZONES)
_N_ZONES = len(ZONE_NAMES)
# World v2 adds the two hunger tanks to the observation, so the fixed block grows by two.
OBS_FIXED = 8 + 2 + 3 * _N_PUBLIC + 1 + _N_ZONES + N_ACTIONS + 1 + TRAIT_VECTOR_LEN
# World v2 inserts the two hunger tanks at indices 3 and 4, directly after E, F and C, because
# they are internal state and belong beside it. Everything after them shifts by two.
OBS = {
    "states": slice(0, 3), "tanks": slice(3, 5), "wealth": 5, "clock": slice(6, 8), "xy": slice(8, 10),
    "dist": slice(10, 10 + _N_PUBLIC),
    "occ": slice(10 + _N_PUBLIC, 10 + 2 * _N_PUBLIC),
    "open": slice(10 + 2 * _N_PUBLIC, 10 + 3 * _N_PUBLIC),
    "home_dist": 10 + 3 * _N_PUBLIC,
    "zone": slice(11 + 3 * _N_PUBLIC, 11 + 3 * _N_PUBLIC + _N_ZONES),
    "prev_action": slice(11 + 3 * _N_PUBLIC + _N_ZONES, 11 + 3 * _N_PUBLIC + _N_ZONES + N_ACTIONS),
    "active": 11 + 3 * _N_PUBLIC + _N_ZONES + N_ACTIONS,
    "traits": slice(12 + 3 * _N_PUBLIC + _N_ZONES + N_ACTIONS, 12 + 3 * _N_PUBLIC + _N_ZONES + N_ACTIONS + TRAIT_VECTOR_LEN),
    "agent_id": slice(OBS_FIXED, None),
}
assert OBS_FIXED == 48 and OBS["active"] == 40 and OBS["zone"] == slice(26, 33)
assert OBS["traits"] == slice(41, 48) and TRAIT_VECTOR_LEN == 7
assert OBS["tanks"] == slice(3, 5)

# ---- log schema: one row per (tick, agent) in every evaluation Parquet file -----------
LOG_COLUMNS = [
    "seed", "condition", "checkpoint", "episode", "t", "day", "t_day", "agent",
    # x and y are held at the agent's zone anchor: position left the simulation with the travel rule and
    # the map is presentation only. "decision" is true on the ticks the policy was consulted;
    # "action" carries the held action while a journey is running.
    "x", "y", "zone", "action", "decision", "active", "queued",
    # World v2: fast_tank and slow_tank are the two hunger tanks and F is their clipped sum;
    # meal_type is "none", "fast" (a market meal) or "slow" (a Food Street tick); company_others
    # is the number of other villagers active at the bar on this tick, staff excluded.
    "E", "F", "C", "fast_tank", "slow_tank", "meal_type", "company_others",
    # rest_idle_cost is the restlessness charge for sitting at home once rested, in drive units.
    # It sits beside effort_penalty because both are reward terms rather than state: D is the
    # drive the agent carries and is what dashboards report, while the reward is
    # -(D) - effort_penalty - rest_idle_cost. Keeping the three columns separate is what makes
    # the reward stream decomposable after the fact.
    # Economy v3: what the tick cost at a counter and at the bar, and whether a drink was bought.
    # meal_type names the source: none, food_street, market or farm.
    "W", "M", "D", "reward", "effort_penalty", "rest_idle_cost",
    "coins_spent_food", "coins_spent_drink", "drank", "informed",
    "logp0", "logp1", "logp2", "logp3", "logp4", "logp5", "logp6",   # NaN for non-learned policies
]
assert len([c for c in LOG_COLUMNS if c.startswith("logp")]) == N_ACTIONS
