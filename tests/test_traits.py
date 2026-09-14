"""Tests of the six-trait system (hamlet/traits.py and its use in the core, policies and plumbing).

The neutral population is covered by tests/test_regression_world.py, which
must keep every jobset-v1 hash; the tests here cover the trait machinery,
the seven shipped variants and the dynamics each trait changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hamlet.config import (
    FARM,
    GO_FARM,
    GO_HOME,
    GREEDY_SERVE_TRIGGER,
    HOME,
    IDLE,
    JOBS,
    LOG_COLUMNS,
    N_ACTIONS,
    OBS,
    OBS_FIXED,
    OFFICE,
    TRAIT_VECTOR_LEN,
    HamletConfig,
)
from hamlet.core import HamletCore
from hamlet.evaluate import evaluate, rollout, trait_and_home_shuffle_counts, traits_path
from hamlet.metrics import dashboard, specialisation
from hamlet.policies import GreedyClockPolicy, GreedyClockWorkPolicy, GreedyStatePolicy
from hamlet.traits import (
    CAPS,
    POPULATIONS,
    SCALAR_TRAITS,
    TraitArrays,
    Traits,
    load_population,
    read_traits_csv,
    sample_traits,
    write_traits_csv,
)

# The condition name carries the population size, so the tests follow it rather than
# pinning a literal: N has already moved once, from 8 to 9.
NN = HamletConfig().n_agents

VARIANTS = tuple(POPULATIONS)
assert len(VARIANTS) == 7


def traits_with(n: int, **overrides) -> TraitArrays:
    """Neutral traits for ``n`` agents with some fields replaced (same value for every agent)."""
    base = Traits.neutral().__dict__ | overrides
    return TraitArrays.from_traits([Traits(**base)] * n)


def within_caps(t: Traits) -> bool:
    ok = all(CAPS[name][0] <= getattr(t, name) <= CAPS[name][1] for name in SCALAR_TRAITS)
    return ok and all(CAPS["aptitude"][0] <= a <= CAPS["aptitude"][1] for a in t.aptitude)


# 1. neutral default and the observation block ---------------------------------------

def test_default_population_is_neutral_and_observation_carries_the_neutral_vector():
    cfg = HamletConfig()
    assert cfg.population == "neutral"
    assert cfg.condition_name == f"A-S1-N{NN}"
    assert HamletConfig(population="hetero_core").condition_name == f"A-S1-N{NN}-pop_hetero_core"
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    block = obs[:, OBS["traits"]]
    assert block.shape == (cfg.N, TRAIT_VECTOR_LEN)
    assert np.array_equal(block, np.tile(Traits.neutral().to_vector(), (cfg.N, 1)))
    assert (core.traits.appetite == 1.0).all() and (core.traits.chronotype == 0.0).all()
    assert (core.traits.aptitude == 1.0).all() and (core.traits.learning_rate == 0.0).all()
    assert core.skill.shape == (cfg.N, len(JOBS)) and (core.skill == 0.0).all()
    hidden = HamletCore(HamletConfig(obs_include_traits=False), 0)
    assert (hidden.reset(0)[:, OBS["traits"]] == 0.0).all()
    assert hidden.reset(0).shape == obs.shape
    with pytest.raises(AssertionError):
        HamletConfig(population="no_such_variant").validate()


# 2. caps -------------------------------------------------------------------------------

def test_hetero_core_draws_stay_within_every_cap():
    draws = sample_traits("hetero_core", 10_000, run_seed=3)
    assert len(draws) == 10_000
    assert all(within_caps(t) for t in draws)
    assert all(0.0 <= t.laziness <= 0.6 for t in draws)
    assert all(t.learning_rate == 0.01 for t in draws)
    # every distribution actually spreads
    for name in ("appetite", "metabolism", "chronotype", "laziness"):
        assert np.std([getattr(t, name) for t in draws]) > 0.05, name


def test_variant_outside_a_cap_fails_at_load():
    with pytest.raises(ValueError):
        load_population({"appetite": {"fixed": 3.0}})
    with pytest.raises(ValueError):
        load_population({"metabolism": {"dist": {"type": "lognormal", "sigma": 0.3, "clip": [0.5, 2.0]}}})
    with pytest.raises(ValueError):
        load_population({"laziness": {"dist": {"type": "uniform", "range": [0.0, 1.5]}}})
    with pytest.raises(ValueError):
        load_population({"aptitude": {"groups": [{"fraction": 0.5, "fixed": [2.5, 0.5]}, {"fraction": 0.5, "fixed": [1, 1]}]}})
    with pytest.raises(ValueError):
        load_population({"chronotype": {"groups": [{"fraction": 0.6, "fixed": 1.0}, {"fraction": 0.6, "fixed": -1.0}]}})
    with pytest.raises(ValueError):
        load_population({"stamina": {"fixed": 1.0}})
    with pytest.raises(ValueError):
        Traits(appetite=2.5, metabolism=1.0, chronotype=0.0, laziness=0.0, aptitude=(1.0, 1.0), learning_rate=0.0).validate()
    with pytest.raises(ValueError):
        HamletCore(HamletConfig(n_agents=2), 0, traits=traits_with(2, metabolism=1.6))
    for name in VARIANTS:
        load_population(name)


# 3. reproducibility and independence from the world stream ---------------------------

def test_traits_reproducible_per_seed_and_agent_and_world_untouched():
    a = sample_traits("hetero_core", 8, run_seed=5)
    b = sample_traits("hetero_core", 12, run_seed=5)
    assert a == b[:8]                                   # (run_seed, agent_id) fixes the traits
    assert a == sample_traits("hetero_core", 8, run_seed=5)
    assert len({t.appetite for t in a}) == 8            # different agents differ
    assert sample_traits("hetero_core", 8, run_seed=6) != a
    for seed in (0, 10_000):
        neutral = HamletCore(HamletConfig(), seed)
        for name in VARIANTS:
            other = HamletCore(HamletConfig(population=name), seed)
            # homes are a fixed layout: identical across variants and seeds
            assert np.array_equal(neutral.home, other.home), name
            for attr in ("E", "F", "C", "informed", "pos"):
                assert np.array_equal(getattr(neutral, attr), getattr(other, attr)), (name, attr)
            assert neutral.rng.bit_generator.state == other.rng.bit_generator.state, name
    # traits_seed defaults to the construction seed and is separate from the world seed
    c1 = HamletCore(HamletConfig(population="hetero_core"), 4)
    c2 = HamletCore(HamletConfig(population="hetero_core"), 9, traits_seed=4)
    assert c1.traits_seed == 4 and np.array_equal(c1.traits.appetite, c2.traits.appetite)
    # homes do not depend on the world seed: there is no per-episode permutation
    assert np.array_equal(c1.home, c2.home)


# 4. aptitude normalisation --------------------------------------------------------------

@pytest.mark.parametrize("name", ["hetero_core", "aptitude_only", "aptitude_split"])
def test_aptitude_geometric_mean_is_one(name):
    for t in sample_traits(name, 16, run_seed=1):
        assert np.exp(np.log(t.aptitude).mean()) == pytest.approx(1.0, abs=1e-6)
    raw = sample_traits("aptitude_only", 16, run_seed=1, normalise=False)
    assert any(abs(np.exp(np.log(t.aptitude).mean()) - 1.0) > 1e-3 for t in raw)
    split = sample_traits("aptitude_split", 8, run_seed=0)
    prefer_farm = [t.aptitude[0] > t.aptitude[1] for t in split]
    assert sum(prefer_farm) == 4
    assert all(t.aptitude[0] == pytest.approx(np.sqrt(2.0)) or t.aptitude[1] == pytest.approx(np.sqrt(2.0)) for t in split)


# helpers for the dynamics tests --------------------------------------------------------

def greedy_clock_run(population: str, seed: int, days: int = 3):
    cfg = HamletConfig(population=population, n_days=days)
    core = HamletCore(cfg, seed)
    obs = core.reset(seed)
    pol = GreedyClockPolicy()
    energy, sleeping = [], []
    for _ in range(cfg.episode_ticks):
        a, _ = pol.act(obs, core)
        obs, _, _ = core.step(a)
        energy.append(core.E.copy())
        sleeping.append(core.active & (core.zone == HOME))
    return cfg, core, np.array(energy), np.array(sleeping)


# 5. metabolism_split ----------------------------------------------------------------------

def test_metabolism_split_heavy_group_has_less_energy_and_sleeps_more():
    heavy_E, light_E, heavy_sleep, light_sleep = [], [], [], []
    for seed in (0, 1, 10_000):
        cfg, core, energy, sleeping = greedy_clock_run("metabolism_split", seed)
        heavy = core.traits.metabolism == CAPS["metabolism"][1]
        assert heavy.sum() == 4 and (core.traits.metabolism[~heavy] == 1.0).all()
        heavy_E.append(energy[:, heavy].mean())
        light_E.append(energy[:, ~heavy].mean())
        heavy_sleep.append(sleeping[:, heavy].sum() / heavy.sum())
        light_sleep.append(sleeping[:, ~heavy].sum() / (~heavy).sum())
    assert np.mean(heavy_E) < np.mean(light_E) - 0.05
    assert np.mean(heavy_sleep) > np.mean(light_sleep)


# 6. chronotype_split ----------------------------------------------------------------------

def sleep_onsets(sleeping: np.ndarray, cfg: HamletConfig) -> dict[int, list[float]]:
    """Per agent and night (noon to noon): start hour of the longest run of active rest at home."""
    total, n = sleeping.shape
    noon = cfg.ticks_per_day // 2
    out: dict[int, list[float]] = {i: [] for i in range(n)}
    for i in range(n):
        for start in range(noon, total - cfg.ticks_per_day + 1, cfg.ticks_per_day):
            seg = sleeping[start:start + cfg.ticks_per_day, i]
            best, best_start, run, run_start = 0, None, 0, 0
            for k, v in enumerate(seg):
                run = run + 1 if v else 0
                if v and run == 1:
                    run_start = k
                if run > best:
                    best, best_start = run, run_start
            if best_start is not None:
                out[i].append(12.0 + best_start / cfg.ticks_per_hour)
    return out


def test_chronotype_split_late_group_falls_asleep_later():
    for seed in (0, 10_000):
        cfg, core, _, sleeping = greedy_clock_run("chronotype_split", seed)
        late = core.traits.chronotype > 0
        assert late.sum() == 4 and set(core.traits.chronotype.tolist()) == {-2.0, 2.0}
        onsets = sleep_onsets(sleeping, cfg)
        late_onsets = [v for i in range(cfg.N) if late[i] for v in onsets[i]]
        early_onsets = [v for i in range(cfg.N) if not late[i] for v in onsets[i]]
        assert late_onsets and early_onsets
        assert np.median(late_onsets) > np.median(early_onsets) + 2.0
    # the window itself: +2 h starts at midnight, -2 h at 20:00
    core = HamletCore(HamletConfig(population="chronotype_split"), 0)
    late = core.traits.chronotype > 0
    assert core.night_window(t_day=230)[~late].all() and not core.night_window(t_day=230)[late].any()
    assert core.night_window(t_day=10)[late].all() and core.night_window(t_day=10)[~late].all()
    assert core.night_window(t_day=70)[late].all() and not core.night_window(t_day=70)[~late].any()
    # GREEDY-STATE stays clock-blind: chronotype never changes its actions
    blind = HamletCore(HamletConfig(population="chronotype_split", n_agents=4), 0)
    blind.t = 230
    blind.E[:] = 0.4          # below GREEDY_SERVE_TRIGGER, so energy is eligible and is the largest deficit
    blind.F[:] = 0.9
    blind.C[:] = 0.9
    a, _ = GreedyStatePolicy().act(blind._observe(), blind)
    assert a.tolist() == [GO_HOME] * 4


# 7. learning ----------------------------------------------------------------------------

def test_learning_only_skill_grows_and_coins_rise_at_the_farm():
    cfg = HamletConfig(population="learning_only", n_agents=2, cap_job=2, n_days=2)
    core = HamletCore(cfg, 0)
    core.reset(0)
    assert (core.traits.learning_rate == 0.02).all()
    go = np.full(2, GO_FARM)
    skills, coins = [], []
    for _ in range(cfg.episode_ticks):
        core.E[:] = 1.0      # productivity pinned to one so that coins track skill alone
        core.F[:] = 1.0
        core.step(go)
        # World v2 pays no wage over the unpaid lunch hour, so those ticks are not evidence about
        # skill and are left out; the seat is kept either way.
        if core.active.all() and cfg.job_pays(core._step_t_day):
            skills.append(core.skill[:, 0].copy())
            coins.append(core.coins_earned.copy())
    skills, coins = np.array(skills), np.array(coins)
    assert len(skills) > 100
    assert (np.diff(skills, axis=0) > 0).all()               # strictly increasing on every active tick
    assert (skills <= cfg.skill_max).all() and (core.skill[:, 1] == 0.0).all()
    assert (np.diff(coins, axis=0) > 0).all()
    # Economy v3: the farm pays farm_wage, a share of the office wage, not the office wage itself.
    assert np.allclose(coins[-1], cfg.farm_wage * (1.0 + skills[-2]))  # the skill before the tick's growth
    assert core.skill[:, 0].max() > 0.9
    # reset clears the skill, the traits stay
    core.reset(0)
    assert (core.skill == 0.0).all() and (core.traits.learning_rate == 0.02).all()
    # forgetting decays the job not worked
    f = HamletCore(HamletConfig(population="learning_only", n_agents=2, cap_job=2, forgetting_rate=0.1), 0)
    f.reset(0)
    f.skill[:] = 0.5
    f.t = cfg.job_open[0]
    for _ in range(f.cfg.travel_ticks + 1):        # travel to the farm, then the arrival tick
        f.step(go)
    f.skill[:] = 0.5                               # the journey itself grew no skill; reset for the check
    f.step(go)
    assert f.active.all()
    assert np.allclose(f.skill[:, 1], 0.45) and (f.skill[:, 0] > 0.5).all()


# 8. observation shape ----------------------------------------------------------------------

def test_observation_shape_identical_across_variants():
    shapes = set()
    for name in VARIANTS:
        cfg = HamletConfig(population=name)
        core = HamletCore(cfg, 0)
        obs = core.reset(0)
        shapes.add(obs.shape)
        assert obs.shape == (cfg.N, OBS_FIXED + cfg.N) == (cfg.N, cfg.obs_dim)
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        assert np.allclose(obs[:, OBS["traits"]], core.traits.to_vectors())
        assert np.allclose(obs[:, OBS["agent_id"]], np.eye(cfg.N))
    assert len(shapes) == 1


# 9. effort penalty ------------------------------------------------------------------------

def test_effort_penalty_zero_at_laziness_zero_and_exact_at_laziness_one():
    cfg = HamletConfig(n_agents=2, cap_job=2)
    neutral = HamletCore(cfg, 0)
    obs = neutral.reset(0)
    pol = GreedyClockPolicy()
    for _ in range(300):
        a, _ = pol.act(obs, neutral)
        obs, rew, _ = neutral.step(a)
        assert np.array_equal(rew, (-neutral.D).astype(np.float32))
        assert (neutral.effort_penalty == 0.0).all()
    lazy = HamletCore(cfg, 0, traits=traits_with(2, laziness=1.0))
    lazy.reset(0)
    assert (lazy.traits.laziness == 1.0).all()
    # travelling to the farm: every TRANSIT tick of the journey costs the coefficient
    for _ in range(cfg.travel_ticks):
        _, rew, _ = lazy.step(np.full(2, GO_FARM))
        assert np.allclose(lazy.effort_penalty, cfg.effort_penalty_coef)
        assert np.allclose(rew, (-lazy.D - cfg.effort_penalty_coef).astype(np.float32))
    lazy.step(np.full(2, GO_FARM))                 # the arrival tick: standing still, no travel cost
    # idling on the spot costs nothing
    _, rew, _ = lazy.step(np.full(2, IDLE))
    assert (lazy.effort_penalty == 0.0).all() and np.allclose(rew, -lazy.D)
    # working costs the coefficient; the log carries the column
    while lazy.zone[0] != FARM:
        lazy.step(np.full(2, GO_FARM))
    while not cfg.job_is_open(lazy.t_day):
        lazy.step(np.full(2, IDLE))
    _, rew, _ = lazy.step(np.full(2, GO_FARM))
    assert lazy.active.all()
    assert np.allclose(lazy.effort_penalty, cfg.effort_penalty_coef)
    cols = lazy.log_columns(0, "x", "none", 0, np.full(2, GO_FARM), rew)
    assert list(cols) == LOG_COLUMNS and "effort_penalty" in LOG_COLUMNS
    assert np.allclose(cols["effort_penalty"], cfg.effort_penalty_coef)
    # income arms scale the coefficient by the wage
    income = HamletCore(HamletConfig(n_agents=2, arm="C", wage=2.0), 0, traits=traits_with(2, laziness=1.0))
    income.reset(0)
    _, rew, _ = income.step(np.full(2, GO_FARM))
    assert np.allclose(income.effort_penalty, cfg.effort_penalty_coef * 2.0)
    assert np.allclose(rew, -cfg.effort_penalty_coef * 2.0)
    # half laziness, half penalty
    half = HamletCore(cfg, 0, traits=traits_with(2, laziness=0.5))
    half.reset(0)
    half.step(np.full(2, GO_FARM))
    assert np.allclose(half.effort_penalty, 0.5 * cfg.effort_penalty_coef)


# 10. greedy act threshold ------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls", [GreedyStatePolicy, GreedyClockPolicy, GreedyClockWorkPolicy])
def test_greedy_threshold_idles_at_laziness_one(policy_cls):
    """Laziness lowers the start trigger, so a lazy agent starts serving a need later.

    A need becomes eligible at ``GREEDY_SERVE_TRIGGER - greedy_threshold_drop * laziness``: 0.5 at
    laziness 0 and 0.2 at laziness 1. The levels below sit between the two, so the same world reads
    as a need worth serving to a diligent agent and as nothing worth leaving the house for to a
    lazy one. Economy v3 introduced the trigger; before it the eligibility ceiling was
    ``greedy_base_threshold`` (1.0), and a level of 0.75 was served at laziness 0 and idled at
    laziness 1. That constant no longer gates anything, and the arithmetic here is the live rule.
    """
    cfg = HamletConfig(n_agents=2)
    level = 0.35                               # below the trigger at laziness 0, above it at laziness 1
    for laziness, expected in ((1.0, IDLE), (0.0, GO_HOME)):
        core = HamletCore(cfg, 0, traits=traits_with(2, laziness=laziness))
        core.reset(0)
        core.t = cfg.job_open[1] + 10          # evening: awake, jobs shut, so no work rule
        assert not cfg.is_night(core.t_day) and not cfg.job_is_open(core.t_day)
        core.E[:] = level
        core.F[:] = level
        core.C[:] = level
        # Above the budget, so economy v3's money rule never fires and the trigger is the only
        # thing deciding whether the agent leaves the house.
        core.W[:] = cfg.daily_budget * cfg.budget_margin + 1.0
        a, _ = policy_cls().act(core._observe(), core)
        assert a.tolist() == [expected] * 2, (policy_cls.__name__, laziness)
    assert GREEDY_SERVE_TRIGGER - cfg.greedy_threshold_drop * 1.0 == pytest.approx(0.2)
    assert GREEDY_SERVE_TRIGGER - cfg.greedy_threshold_drop * 0.0 == pytest.approx(0.5)
    # A level below the lowered threshold is still served: laziness raises the bar, it does not
    # switch the need off. Energy and social stay above 0.2 so satiety is the only eligible need,
    # and in job hours economy v3 serves satiety at the farm, which feeds while it pays.
    core = HamletCore(cfg, 0, traits=traits_with(2, laziness=1.0))
    core.reset(0)
    core.t = cfg.job_open[0] + 10
    core.E[:] = 0.75
    core.F[:] = 0.15                           # below 0.2, the trigger at laziness 1
    core.C[:] = 0.75
    a, _ = GreedyClockPolicy().act(core._observe(), core)
    assert (a != IDLE).all() and (a != GO_HOME).all()


# plumbing ----------------------------------------------------------------------------------

def test_csv_round_trip_and_evaluate_writes_traits(tmp_path):
    traits = sample_traits("hetero_core", 6, run_seed=2)
    path = write_traits_csv(tmp_path / "traits.csv", traits, "hetero_core")
    back, variant = read_traits_csv(path)
    assert variant == "hetero_core" and len(back) == 6
    for x, y in zip(traits, back):
        assert x.appetite == pytest.approx(y.appetite) and x.aptitude == pytest.approx(y.aptitude)
    df = pd.read_csv(path)
    assert list(df.columns) == ["agent_id", "variant", "appetite", "metabolism", "chronotype", "laziness",
                                "learning_rate", "aptitude_FARM", "aptitude_OFFICE"]

    cfg = HamletConfig(n_agents=4, n_days=2, population="aptitude_split")
    paths = evaluate(cfg, GreedyClockWorkPolicy(), [10_000], tmp_path)
    assert paths[0].parent.name == "GREEDY-CLOCK-WORK" and paths[0].parent.parent.name == cfg.condition_name
    csv_path = traits_path(tmp_path, cfg.condition_name, "GREEDY-CLOCK-WORK", 10_000)
    assert csv_path.exists()
    back, variant = read_traits_csv(csv_path)
    assert variant == "aptitude_split"
    assert [t.aptitude for t in back] == [t.aptitude for t in sample_traits("aptitude_split", 4, 10_000)]
    log = pd.read_parquet(paths[0])
    assert list(log.columns) == LOG_COLUMNS and log["effort_penalty"].dtype == np.float32
    summary = dashboard.summary(paths[0].parent, cfg.burn_in_days)
    assert summary["population"].iloc[0] == "aptitude_split"
    assert summary["mean_aptitude_FARM"].iloc[0] == pytest.approx(np.mean([t.aptitude[0] for t in back]))
    assert dashboard.summary(log)["population"].iloc[0] == "aptitude_split"


def test_greedy_clock_work_uses_the_aptitude_trait():
    cfg = HamletConfig(n_agents=4, population="aptitude_split")
    core = HamletCore(cfg, 0)
    core.reset(0)
    core.t = cfg.job_open[0]
    core.E[:] = 1.0
    core.F[:] = 1.0
    core.C[:] = 1.0
    a, _ = GreedyClockWorkPolicy().act(core._observe(), core)
    prefer_farm = core.traits.aptitude[:, 0] > core.traits.aptitude[:, 1]
    assert (a[prefer_farm] == GO_FARM).all() and (a[~prefer_farm] != GO_FARM).all()
    override = np.ones((4, len(JOBS)))
    override[:, 1] = 2.0
    b, _ = GreedyClockWorkPolicy(aptitude=override).act(core._observe(), core)
    assert (b != GO_FARM).all()


def test_swap_test_kinds_and_trait_and_home_shuffle_null():
    n = 4
    cfg = HamletConfig(n_agents=n, n_days=2, population="hetero_core")
    core = HamletCore(cfg, 0)
    obs = core.reset(0)
    vectors = core.traits.to_vectors()

    def keyed_on_traits(o: np.ndarray) -> np.ndarray:
        out = np.zeros((o.shape[0], N_ACTIONS))
        out[np.arange(o.shape[0]), (o[:, OBS["traits"]].sum(axis=1) * 1000).astype(int) % N_ACTIONS] = 1.0
        return out

    def keyed_on_id(o: np.ndarray) -> np.ndarray:
        out = np.zeros((o.shape[0], N_ACTIONS))
        out[np.arange(o.shape[0]), o[:, OBS["agent_id"]].argmax(axis=1)] = 1.0
        return out

    assert specialisation.swap_test(keyed_on_traits, obs, n, swap="id") == pytest.approx(0.0)
    assert specialisation.swap_test(keyed_on_traits, obs, n, swap="traits", trait_vectors=vectors) > 0.0
    assert specialisation.swap_test(keyed_on_id, obs, n, swap="traits", trait_vectors=vectors) == pytest.approx(0.0)
    assert specialisation.swap_test(keyed_on_id, obs, n, swap="both", trait_vectors=vectors) > 0.0
    with pytest.raises(ValueError):
        specialisation.swap_test(keyed_on_id, obs, n, swap="traits")
    with pytest.raises(ValueError):
        specialisation.swap_test(keyed_on_id, obs, n, swap="homes")

    df = rollout(cfg, GreedyClockWorkPolicy(), 10_000, 0)
    counts = specialisation.time_budget(df, burn_in_days=cfg.burn_in_days, n_agents=n)
    with pytest.raises(ValueError):
        specialisation.dol_indiv(counts, null="trait_and_home_shuffle")
    replays = trait_and_home_shuffle_counts(cfg, GreedyClockWorkPolicy, 10_000, n_perm=3, rng=np.random.default_rng(1))
    assert len(replays) == 3 and all(m.shape == counts.shape for m in replays)
    res = specialisation.dol_indiv(counts, null="trait_and_home_shuffle", replay_counts=replays)
    assert set(res) == {"value", "null_mean", "null_sd", "z"}
    # the identity permutation reproduces the observed run exactly
    same = rollout(cfg, GreedyClockWorkPolicy(), 10_000, 0, identity_perm=np.arange(n))
    assert same.equals(df)
    perm = np.array([1, 0, 3, 2])
    swapped = HamletCore(cfg, 10_000)
    swapped.set_traits(swapped.traits.permuted(perm))
    swapped.reset(10_000)
    plain = HamletCore(cfg, 10_000)
    plain.reset(10_000)
    # HOME is the agent's own zone, so the shuffle null permutes traits alone.
    assert np.array_equal(swapped.home, plain.home)
    assert np.array_equal(swapped.traits.appetite, plain.traits.appetite[perm])
    assert np.array_equal(swapped.E, plain.E)
