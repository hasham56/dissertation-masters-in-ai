"""Unit tests for the metric modules on synthetic logs.

No simulation is run: every test builds a DataFrame in ``LOG_COLUMNS`` with
known structure and checks that the metric recovers it, or that a null
recovers zero where nothing was planted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hamlet.config import CANTEEN, FARM, HOME, LOG_COLUMNS, MARKET, SOCIAL, TRANSIT, ZONE_NAMES, N_ACTIONS, OBS, OBS_FIXED
from hamlet.metrics import dashboard, routine, social, specialisation
from hamlet.metrics.common import group_codes

T = 240
N = 8
DAYS = 4
BURN_IN = 1


# ---------------------------------------------------------------------------
# synthetic log builder
# ---------------------------------------------------------------------------

from hamlet.metrics.synthetic import make_log  # noqa: E402


def markov_chain(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    cum = P.cumsum(axis=1)
    u = rng.random(n)
    s = np.empty(n, dtype=np.int64)
    s[0] = rng.integers(P.shape[0])
    for i in range(1, n):
        s[i] = np.searchsorted(cum[s[i - 1]], u[i])
    return s


def entropy_rate(P: np.ndarray) -> float:
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    with np.errstate(divide="ignore"):
        logs = np.where(P > 0, np.log2(np.where(P > 0, P, 1.0)), 0.0)
    return float(-(pi[:, None] * P * logs).sum())


# ---------------------------------------------------------------------------
# routine
# ---------------------------------------------------------------------------

def test_lz_entropy_rate_markov_chain_within_10_percent():
    P = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    seq = markov_chain(P, 5000, np.random.default_rng(1))
    est = routine.lz_entropy_rate(seq)
    true = entropy_rate(P)
    assert abs(est - true) / true < 0.10


def test_lz_entropy_rate_iid_within_10_percent():
    p = np.array([0.5, 0.3, 0.2])
    seq = np.random.default_rng(2).choice(3, size=5000, p=p)
    est = routine.lz_entropy_rate(seq)
    true = float(-(p * np.log2(p)).sum())
    assert abs(est - true) / true < 0.10
    assert routine.uncorrelated_entropy(seq) == pytest.approx(true, abs=0.05)
    assert routine.random_entropy(seq) == pytest.approx(np.log2(3))


def test_lz_constant_sequence_is_near_zero():
    assert routine.lz_entropy_rate(np.zeros(500, dtype=int)) < 0.1


def test_fano_predictability_inverts_the_bound():
    n = 6
    for pi in (0.3, 0.6, 0.9):
        s = -pi * np.log2(pi) - (1 - pi) * np.log2(1 - pi) + (1 - pi) * np.log2(n - 1)
        assert routine.fano_predictability(s, n) == pytest.approx(pi, abs=1e-6)
    assert routine.fano_predictability(0.0, n) == 1.0
    assert routine.fano_predictability(np.log2(n), n) == pytest.approx(1.0 / n)
    assert routine.fano_predictability(1.0, n) > routine.fano_predictability(2.0, n)


def test_add_bins_codes():
    rng = np.random.default_rng(3)
    df = routine.add_bins(make_log(rng), job_open=(80, 180), market_open=(60, 200), hour_bin_ticks=30)
    assert set(df["hour_bin"].unique()) == set(range(8))
    assert df["state_bin"].between(0, 26).all()
    assert set(df["open_code"].unique()) == {0, 1, 3}
    td = df["t_day"].to_numpy()
    assert (df.loc[(td >= 80) & (td < 180), "open_code"] == 3).all()
    assert (df.loc[td < 60, "open_code"] == 0).all()
    assert set(df["night"].unique()) == {0, 1}
    assert (df["night"].to_numpy() == ((td >= 220) | (td < 60))).all()
    custom = routine.add_bins(make_log(rng), night=(200, 40))
    assert (custom["night"].to_numpy() == ((td >= 200) | (td < 40))).all()


def test_conditional_mi_null_on_independent_data():
    """Both nulls are neutral on independent data, judged over several seeds rather than one.

    A single seed cannot test this: a correct null puts |z| above 2 about one time in twenty, so a
    one-seed bound fails at that rate for no reason. Measured over 40 seeds the block null has mean
    z -0.082 and sd 0.963, which is what neutrality looks like; seed 4 alone reads +2.216.
    """
    block, relabel = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        df = make_log(rng)
        res = routine.conditional_mi(df, n_perm=100, rng=rng, null="block")
        assert set(res) == {"value", "null_mean", "null_sd", "z"}
        block.append(res["z"])
        relabel.append(routine.conditional_mi(df, n_perm=100, rng=rng)["z"])
    for name, zs in (("block", block), ("relabel", relabel)):
        assert abs(float(np.mean(zs))) < 1.0, (name, zs)
        assert sum(abs(z) > 2.0 for z in zs) <= 1, (name, zs)


def test_conditional_mi_detects_clock_dependence():
    rng = np.random.default_rng(5)
    base = make_log(rng)
    zone = (base["t_day"].to_numpy() // 30) % 6
    df = make_log(rng, zone=zone)
    res = routine.conditional_mi(df, n_perm=100, rng=rng, null="block")
    assert res["value"] > 1.0
    assert res["z"] > 5.0


def test_conditional_mi_state_given_hour():
    rng = np.random.default_rng(6)
    base = routine.add_bins(make_log(rng))
    zone = base["state_bin"].to_numpy() % 6
    df = make_log(rng, zone=zone, states=base[["E", "F", "C"]].to_numpy())
    res = routine.conditional_mi(df, given=("state_bin",), cond=("hour_bin", "open_code"), n_perm=50, rng=rng, null="block")
    assert res["z"] > 5.0


def test_regularity_one_for_fixed_schedule_and_low_for_random():
    rng = np.random.default_rng(7)
    base = make_log(rng)
    fixed = make_log(rng, zone=(base["t_day"].to_numpy() // 30) % 6)
    assert routine.regularity(fixed, ticks_per_day=T, slot=30) == pytest.approx(1.0)
    assert routine.regularity(fixed, ticks_per_day=T, slot=40) < 1.0
    random = make_log(rng)
    assert routine.regularity(random, ticks_per_day=T) < 0.4


def test_periodogram_power_at_day_and_within_day_shuffle():
    rng = np.random.default_rng(8)
    base = make_log(rng)
    zone = np.where(base["t_day"].to_numpy() < T // 2, FARM, HOME)
    df = make_log(rng, zone=zone)
    frac = routine.periodogram_power_at_day(df, FARM, ticks_per_day=T)
    assert frac > 0.5
    per_agent = routine.periodogram_power_at_day(df, FARM, ticks_per_day=T, per_agent=True)
    assert per_agent.shape == (N,)
    shuffled = routine.within_day_shuffle(df, rng)
    assert routine.periodogram_power_at_day(shuffled, FARM, ticks_per_day=T) < 0.2
    # composition of every agent-day is preserved
    before = df.groupby(["agent", "day"])["zone"].sum()
    after = shuffled.groupby(["agent", "day"])["zone"].sum()
    assert before.equals(after)


def test_actogram_matrix_shape_and_values():
    rng = np.random.default_rng(9)
    base = make_log(rng)
    zone = (base["t_day"].to_numpy() // 40) % 6
    df = make_log(rng, zone=zone)
    m = routine.actogram_matrix(df, agent=3, ticks_per_day=T)
    assert m.shape == (DAYS, T)
    assert (m[:, 0] == 0).all() and (m[:, 239] == 5).all()


def _schedule_log(rng, shock_day=None, shift=0, persistent=False, d_level=0.5, d_shock=1.0, n_days=6):
    base = make_log(rng, n_days=n_days)
    t_day = base["t_day"].to_numpy()
    day = base["day"].to_numpy()
    zone = ((t_day + shift * (day == shock_day)) // 40) % 6
    e = np.full(len(base), 1 - np.sqrt(d_level / 3))
    if shock_day is not None:
        hit = (day >= shock_day) if persistent else (day == shock_day)
        e = np.where(hit, 1 - np.sqrt(d_shock / 3), e)
    states = np.stack([e, e, e], axis=1)
    return make_log(rng, n_days=n_days, zone=zone, states=states)


def test_shock_recovery_transient_and_persistent():
    rng = np.random.default_rng(10)
    control = _schedule_log(rng)
    transient = _schedule_log(rng, shock_day=4, shift=20)
    res = routine.shock_recovery(transient, control, shock_day=4, ticks_per_day=T, burn_in_days=BURN_IN)
    # Trailing daily mean returns within 10% once 90% of the shock day has left the window.
    # recovery_ticks is the discrete tick at which a continuous trailing mean crosses a threshold,
    # so it resolves only to the nearest tick: the same input gives 216 on an i5-10300H and 215 on
    # a GCP n2 with the identical pinned numpy. The crossing is the claim, not
    # the exact tick, so the tolerance is one tick.
    assert res["recovery_ticks"] == pytest.approx(0.9 * T, abs=1.0)
    assert res["jaccard_pre"] == pytest.approx(1.0)
    assert res["jaccard_shock"] < 1.0
    assert res["jaccard_post"] == pytest.approx(1.0)
    assert res["jaccard_pre_control"] == pytest.approx(1.0)
    persistent = _schedule_log(rng, shock_day=4, persistent=True)
    res2 = routine.shock_recovery(persistent, control, shock_day=4, ticks_per_day=T, burn_in_days=BURN_IN)
    assert np.isnan(res2["recovery_ticks"])
    # the day-after drive gap: the transient shock leaves none, the persistent one leaves 0.5 drive units
    assert res["drive_gap_post"] == pytest.approx(0.0, abs=1e-9)
    assert res["drive_gap_shock"] == pytest.approx(0.5)
    assert res2["drive_gap_post"] == pytest.approx(0.5)


def _reforming_log(rng, reforms: bool, shock_day=4, shift=20, stay_prob=0.7, n_days=6):
    """A routine with a fixed daily pattern, a shifted shock day, and a day after that either
    returns to the pattern (``reforms``) or stays scrambled: each post-shock tick keeps the old
    pattern with probability ``stay_prob`` and otherwise takes a random zone."""
    base = make_log(rng, n_days=n_days)
    t_day = base["t_day"].to_numpy()
    day = base["day"].to_numpy()
    zone = ((t_day + shift * (day == shock_day)) // 40) % 6
    if not reforms:
        after = day == shock_day + 1
        noise = rng.integers(0, 6, size=len(zone))
        keep = rng.random(len(zone)) < stay_prob
        zone = np.where(after & ~keep, noise, zone)
    return make_log(rng, n_days=n_days, zone=zone)


def test_shock_reformation_separates_a_routine_that_re_forms_from_one_that_does_not():
    rng = np.random.default_rng(11)
    control = _reforming_log(rng, reforms=True, shift=0)
    reformed = _reforming_log(rng, reforms=True)
    scrambled = _reforming_log(rng, reforms=False)
    good = routine.shock_recovery(reformed, control, shock_day=4, ticks_per_day=T, burn_in_days=BURN_IN)
    bad = routine.shock_recovery(scrambled, control, shock_day=4, ticks_per_day=T, burn_in_days=BURN_IN)
    # the control's ordinary consecutive days are identical, so the reference is 1
    assert good["jaccard_ref_control"] == pytest.approx(1.0)
    assert bad["jaccard_ref_control"] == pytest.approx(1.0)
    # both shocked runs lose the pattern on the shock day
    assert good["jaccard_shock"] < 0.5 and bad["jaccard_shock"] < 0.5
    # day 5 re-forms in one run (reformation 0) and not in the other (well below the -0.1 reading)
    assert good["reformation"] == pytest.approx(0.0, abs=1e-9)
    assert bad["reformation"] < -0.1
    assert good["reformation"] - bad["reformation"] > 0.3
    # a day that keeps 70% of its ticks has Jaccard m / (2T - m) with m about 0.7 T plus chance matches
    assert 0.4 < bad["jaccard_post"] < 0.8


# ---------------------------------------------------------------------------
# specialisation
# ---------------------------------------------------------------------------

def test_time_budget_counts():
    rng = np.random.default_rng(11)
    df = make_log(rng)
    counts = specialisation.time_budget(df, burn_in_days=BURN_IN)
    assert counts.shape == (N, len(ZONE_NAMES))
    assert counts.sum() == N * (DAYS - BURN_IN) * T
    assert (counts.sum(axis=1) == (DAYS - BURN_IN) * T).all()


def test_dol_identical_and_disjoint_rows():
    rng = np.random.default_rng(12)
    identical = np.tile([100, 50, 30, 20, 10, 5], (N, 1))
    res = specialisation.dol_indiv(identical, n_perm=50, rng=rng, null="multinomial")
    assert res["value"] == pytest.approx(0.0, abs=1e-12)
    assert specialisation.dol_task_value(identical) == pytest.approx(0.0, abs=1e-12)
    disjoint = np.eye(6, dtype=int) * 100
    res = specialisation.dol_indiv(disjoint, n_perm=50, rng=rng, null="multinomial")
    assert res["value"] == pytest.approx(1.0)
    assert res["z"] > 3.0
    assert specialisation.dol_task_value(disjoint) == pytest.approx(1.0)


def test_dol_multinomial_null_is_centred_when_rows_are_draws_from_the_pool():
    rng = np.random.default_rng(13)
    p = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    counts = rng.multinomial(720, p, size=N)
    res = specialisation.dol_indiv(counts, n_perm=200, rng=rng, null="multinomial")
    assert abs(res["z"]) < 2.5
    with pytest.raises(ValueError):
        specialisation.dol_indiv(counts, n_perm=10, rng=rng)


def test_dol_replay_matrices_are_the_null():
    rng = np.random.default_rng(25)
    p = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    counts = rng.multinomial(720, p, size=N)
    replays = [rng.multinomial(720, p, size=N) for _ in range(30)]
    res = specialisation.dol_indiv(counts, rng=rng, replay_counts=replays)
    expected = np.array([specialisation.dol_indiv_value(m) for m in replays])
    assert res["null_mean"] == pytest.approx(expected.mean())
    assert res["null_sd"] == pytest.approx(expected.std(ddof=1))


def test_time_budget_per_day_sums_to_the_flat_budget():
    rng = np.random.default_rng(26)
    df = make_log(rng)
    flat = specialisation.time_budget(df, burn_in_days=BURN_IN)
    per_day = specialisation.time_budget(df, burn_in_days=BURN_IN, per_day=True)
    assert per_day.shape == (N, DAYS - BURN_IN, len(ZONE_NAMES))
    assert (per_day.sum(axis=1) == flat).all()
    assert (per_day.sum(axis=2) == T).all()
    active = rng.random(len(df)) < 0.5
    budget = specialisation.time_budget(make_log(rng, active=active), burn_in_days=BURN_IN, per_day=True, active_only=True)
    assert budget.shape == (N, DAYS - BURN_IN, len(ZONE_NAMES))
    assert budget.sum() == active[BURN_IN * T * N:].sum()


def test_specialisation_index_bounds():
    rng = np.random.default_rng(14)
    identical = np.tile([100, 50, 30, 20, 10, 5], (N, 1))
    assert specialisation.specialisation_index(identical) == pytest.approx(0.0, abs=1e-12)
    disjoint = np.eye(6, dtype=int) * 100
    assert specialisation.specialisation_index(disjoint) == pytest.approx(1.0)
    random = rng.integers(0, 100, size=(N, 6))
    si = specialisation.specialisation_index(random)
    assert 0.0 <= si <= 1.0


def test_pairwise_jsd_properties():
    counts = np.array([[10, 0, 0], [0, 10, 0], [5, 5, 0]])
    m = specialisation.pairwise_jsd(counts)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)
    assert m[0, 1] == pytest.approx(1.0)
    assert 0 < m[0, 2] < 1


def test_cluster_agents_recovers_two_groups():
    rng = np.random.default_rng(15)
    workers = rng.multinomial(720, [0.2, 0.6, 0.1, 0.05, 0.025, 0.025], size=4)
    socials = rng.multinomial(720, [0.2, 0.05, 0.1, 0.6, 0.025, 0.025], size=4)
    counts = np.vstack([workers, socials])
    labels, score = specialisation.cluster_agents(counts, max_k=4)
    truth = np.array([0] * 4 + [1] * 4)
    assert specialisation.ari(labels, truth) == pytest.approx(1.0)
    assert score > 0.5


def test_ari_identical_permuted_and_random():
    rng = np.random.default_rng(16)
    a = rng.integers(0, 4, size=2000)
    assert specialisation.ari(a, a) == pytest.approx(1.0)
    assert specialisation.ari(a, (a + 1) % 4) == pytest.approx(1.0)
    b = rng.integers(0, 4, size=2000)
    assert abs(specialisation.ari(a, b)) < 0.05
    assert specialisation.ari_across_episodes([a, a, a]) == pytest.approx(1.0)


def test_add_bins_fixed_edges_ignore_the_reference():
    rng = np.random.default_rng(19)
    df = make_log(rng)
    skewed = df.copy()
    skewed[["E", "F", "C"]] = skewed[["E", "F", "C"]] ** 4   # a very different reference distribution
    fixed = routine.add_bins(df, ref=skewed, state_edges="fixed")
    fixed_own = routine.add_bins(df, state_edges="fixed")
    assert fixed["state_bin"].equals(fixed_own["state_bin"])
    quant = routine.add_bins(df, ref=skewed)
    assert not quant["state_bin"].equals(fixed["state_bin"])
    # absolute edges at 1/3 and 2/3: a row with every state in [2/3, 1) codes 26, in [0, 1/3) codes 0
    row = df.iloc[[0]].copy()
    row[["E", "F", "C"]] = 0.9
    assert int(routine.add_bins(row, state_edges="fixed")["state_bin"].iloc[0]) == 26
    row[["E", "F", "C"]] = 0.1
    assert int(routine.add_bins(row, state_edges="fixed")["state_bin"].iloc[0]) == 0
    row[["E", "F", "C"]] = [1 / 3, 0.5, 0.0]
    assert int(routine.add_bins(row, state_edges="fixed")["state_bin"].iloc[0]) == 9 + 3 + 0
    with pytest.raises(ValueError):
        routine.add_bins(df, state_edges="thirds")


def test_ari_bootstrap_resamples_episodes_not_pairs():
    rng = np.random.default_rng(18)
    truth = np.array([0] * 4 + [1] * 4)
    # 32 episodes: the same two roles recovered in 24, noise in 8
    episodes = [truth if k < 24 else rng.integers(0, 2, size=8) for k in range(32)]
    res = specialisation.ari_across_episodes_ci(episodes, n_boot=300, rng=rng)
    assert res["value"] == pytest.approx(specialisation.ari_across_episodes(episodes))
    assert res["n_episodes"] == 32
    assert res["ci_low"] <= res["value"] <= res["ci_high"]
    assert res["ci_low"] < res["ci_high"]
    # every episode identical: the interval collapses on 1, and duplicate pairs are not counted
    same = specialisation.ari_across_episodes_ci([truth] * 6, n_boot=50, rng=rng)
    assert same["value"] == 1.0 and same["ci_low"] == 1.0 and same["ci_high"] == 1.0
    # the episode bootstrap is wider than a pair bootstrap on the same dependent data: pairs that
    # share an episode move together, which the pair resample treats as independent information
    m = len(episodes)
    pair_vals = np.array([specialisation.ari(episodes[i], episodes[j]) for i in range(m) for j in range(i + 1, m)])
    pair_draws = np.array([pair_vals[rng.integers(0, pair_vals.size, size=pair_vals.size)].mean() for _ in range(300)])
    pair_width = np.quantile(pair_draws, 0.975) - np.quantile(pair_draws, 0.025)
    assert res["ci_high"] - res["ci_low"] > pair_width
    assert np.isnan(specialisation.ari_across_episodes_ci([truth], n_boot=10, rng=rng)["ci_low"])


def test_swap_test_zero_when_identity_ignored_and_positive_otherwise():
    rng = np.random.default_rng(17)
    obs = rng.random((32, OBS_FIXED + N)).astype(np.float32)

    def blind(o: np.ndarray) -> np.ndarray:
        return np.full((o.shape[0], N_ACTIONS), 1.0 / N_ACTIONS)

    def keyed(o: np.ndarray) -> np.ndarray:
        ids = o[:, OBS["agent_id"]].argmax(axis=1)
        out = np.zeros((o.shape[0], N_ACTIONS))
        out[np.arange(o.shape[0]), ids % N_ACTIONS] = 1.0
        return out

    assert specialisation.swap_test(blind, obs, N) == pytest.approx(0.0)
    mean, matrix = specialisation.swap_test(keyed, obs, N, return_matrix=True)
    assert mean > 0.5
    assert matrix[0, N_ACTIONS] == pytest.approx(0.0)   # ids 0 and N_ACTIONS map to the same action
    assert matrix[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# permutation nulls on autocorrelated data
# ---------------------------------------------------------------------------
#
# The walkers below stand in for RANDOM in the real world. Each agent follows
# a Markov chain over the six zones: it stays put with probability STAY (mean
# dwell 1 / (1 - STAY) = 20 ticks, the time a random walker needs to cross a
# zone) and otherwise redraws a zone from its own propensity vector. The
# propensity is tilted towards the zone next to the agent's home, zone
# i mod 6 for agent i, by HOME_TILT (weight 1 + tilt against 1 for the rest),
# so zone budgets depend on identity through geometry alone, and only mildly:
# the tilt is small against the day-to-day variation of a budget built from
# 240 / 20 = 12 dwell episodes per day. States follow reflected random walks
# with step STATE_STEP per tick, independent of the clock and of the zones, so
# the state terciles form long runs like the real states do. Nothing in the
# construction depends on the clock or on the state, so the conditional
# clock information and the identity-linked division of labour are both zero
# and a calibrated null should put both near z = 0; the tick-shuffle and
# pooled-multinomial nulls treat every tick as an independent draw and score
# both far above 5.

STAY = 0.95
HOME_TILT = 0.3
STATE_STEP = 0.02


def _markov_walkers(rng: np.random.Generator, propensity: np.ndarray, stay: float = STAY) -> np.ndarray:
    """(DAYS * T, N) zone sequences; row i of ``propensity`` is agent i's redraw distribution."""
    total = DAYS * T
    zone = np.empty((total, N), dtype=np.int64)
    for i in range(N):
        p = propensity[i] / propensity[i].sum()
        draws = rng.choice(6, size=total, p=p)
        move = rng.random(total) > stay
        move[0] = True
        idx = np.maximum.accumulate(np.where(move, np.arange(total), 0))
        zone[:, i] = draws[idx]
    return zone


def _smooth_states(rng: np.random.Generator, step: float = STATE_STEP) -> np.ndarray:
    """(DAYS * T, N, 3) reflected random walks in [0, 1], one per agent and need."""
    x = rng.random((1, N, 3)) + np.cumsum(rng.normal(0.0, step, (DAYS * T, N, 3)), axis=0)
    x = np.abs(x) % 2.0
    return np.where(x > 1.0, 2.0 - x, x)


def _home_tilted_walkers(rng: np.random.Generator, tilt: float = HOME_TILT) -> pd.DataFrame:
    propensity = np.ones((N, 6))
    propensity[np.arange(N), np.arange(N) % 6] += tilt
    return make_log(rng, zone=_markov_walkers(rng, propensity), states=_smooth_states(rng))


def test_autocorrelated_walkers_are_null_under_day_shuffle_and_block_nulls():
    rng = np.random.default_rng(27)
    df = _home_tilted_walkers(rng)
    per_day = specialisation.time_budget(df, burn_in_days=BURN_IN, per_day=True)
    old = specialisation.dol_indiv(per_day, n_perm=100, rng=rng, null="multinomial")
    new = specialisation.dol_indiv(per_day, n_perm=100, rng=rng)
    assert old["z"] > 5.0
    assert abs(new["z"]) < 3.0
    assert new["null_sd"] > old["null_sd"]
    tick = routine.conditional_mi(df, n_perm=50, rng=rng, null="tick")
    block = routine.conditional_mi(df, n_perm=50, rng=rng, null="block")
    assert tick["z"] > 5.0
    assert abs(block["z"]) < 3.0
    assert block["null_sd"] > tick["null_sd"]


def test_day_shuffle_null_detects_identity_linked_division_of_labour():
    # agent i spends 60% of its time in zone i mod 6, every day, the rest spread evenly
    rng = np.random.default_rng(28)
    propensity = np.full((N, 6), 0.08)
    propensity[np.arange(N), np.arange(N) % 6] = 0.60
    df = make_log(rng, zone=_markov_walkers(rng, propensity), states=_smooth_states(rng))
    per_day = specialisation.time_budget(df, burn_in_days=BURN_IN, per_day=True)
    res = specialisation.dol_indiv(per_day, n_perm=100, rng=rng)
    assert res["value"] > 0.2
    assert res["z"] > 5.0


def test_block_null_detects_clock_entrained_pattern():
    # the zone is a function of the hour bin and of nothing else
    rng = np.random.default_rng(29)
    base = make_log(rng)
    zone = (base["t_day"].to_numpy() // 30) % 6
    df = make_log(rng, zone=zone, states=_smooth_states(rng))
    res = routine.conditional_mi(df, n_perm=50, rng=rng, null="block")
    assert res["value"] > 1.0
    assert res["z"] > 5.0


def test_block_null_keeps_the_hour_open_relation_and_the_strata():
    rng = np.random.default_rng(30)
    df = _home_tilted_walkers(rng)
    binned = routine.add_bins(df)
    # with open_code left in place while the hour bins move, every stratum would
    # see hour bins it never contains; the block null moves the two together
    res = routine.conditional_mi(binned, n_perm=30, rng=rng, null="block")
    assert np.isfinite(res["z"])
    with pytest.raises(ValueError):
        routine.conditional_mi(binned, n_perm=5, rng=rng, null="row")


def test_night_moves_with_the_clock_under_the_block_null():
    rng = np.random.default_rng(34)
    binned = routine.add_bins(make_log(rng))
    assert "night" in routine._CLOCK_COLUMNS
    blocks = group_codes(binned, ("agent", "day"))
    src = routine._block_shift_index(blocks, binned["t"].to_numpy(), rng)
    hb = binned["hour_bin"].to_numpy()
    night = binned["night"].to_numpy()
    real = np.bincount(hb * 2 + night, minlength=16)
    shifted = np.bincount(hb[src] * 2 + night[src], minlength=16)
    assert (shifted == real).all()
    assert (hb[src] != hb).any()
    # had night stayed in place while the hours moved, hour bins would land in the wrong night state
    assert (np.bincount(hb[src] * 2 + night, minlength=16) != real).any()


# ---------------------------------------------------------------------------
# a schedule-only process: the relabel null against the block null
# ---------------------------------------------------------------------------
#
# Each agent carries three states (E, F, C) that are leaky integrators of its
# own zone history: E integrates time at HOME, F time at CANTEEN, C time at
# SOCIAL, with leak SCHED_LEAK per tick and gain SCHED_GAIN while the serving
# zone is occupied, clipped to [0, 1]. With probability 1 - SCHED_STICK the
# agent redraws its zone, otherwise it stays, so runs last about 10-30 ticks.
# The redraw reads the state, the opening hours and the night flag and nothing
# else: the lowest need is served when it is below SCHED_LOW; otherwise HOME
# at night, FARM while the jobs are open, MARKET while the market is open, each
# with probability SCHED_P, else a zone drawn from SCHED_FALLBACK. No rule
# reads the hour inside a window, so the clock information beyond the
# schedule is zero by construction. The process stands in for GREEDY-CLOCK.
#
# The block null shifts the whole clock of each agent-day, which breaks the
# lock between the schedule and the state trajectory; the shifted data
# occupies more (stratum, hour) cells than the real data, carries more
# plug-in bias, and the null mean lands above the observed value, so a
# schedule-following process scores z below -3 without using the hour at
# all. The relabel null permutes the hour labels inside each agent-day and
# schedule stratum, keeps that lock and scores close to zero. Its neutrality
# needs the hour not to be locked to the state terciles inside a window
# beyond what the schedule imposes: a slower integrator with a global phase
# (every agent rested at night, every state decaying in step through the
# work window) pushes the relabel z to -4 and below on the same rule, which
# is the regime the conditioning set cannot separate from a real dependence.
# The three baselines on the evaluation logs sit in the neutral regime, and
# the margins below were checked on eight seeds.

from hamlet.metrics.synthetic import (  # noqa: E402
    PLANT_BIN, PLANT_PROB, SCHED_DAYS, SCHED_SEEDS, schedule_only_log as _schedule_only_log,
)




@pytest.mark.parametrize("seed", SCHED_SEEDS)
def test_schedule_only_process_is_null_under_relabel_and_biased_under_block(seed):
    df = _schedule_only_log(seed)
    runs = np.diff(np.flatnonzero(np.r_[True, np.diff(df.loc[df["agent"] == 0, "zone"].to_numpy()) != 0, True]))
    assert 8.0 < runs.mean() < 30.0
    relabel = routine.conditional_mi(df, n_perm=100, rng=np.random.default_rng(seed))
    block = routine.conditional_mi(df, n_perm=100, rng=np.random.default_rng(seed), null="block")
    assert abs(relabel["z"]) < 3.0
    # World v2 widened the job window from 100 ticks (08:00-18:00) to 120 (07:00-19:00), so the
    # schedule-only process spreads over more of the day and the block null's plug-in bias is a
    # little weaker. Measured over SCHED_SEEDS under the wider window: -2.986, -4.332, -3.219, so
    # the bar moves from -3.0 to -2.5. The conclusion is unchanged: the block null reads negative
    # on a process that uses no within-window clock information at all.
    assert block["z"] < -2.5
    assert block["null_mean"] > relabel["null_mean"]


@pytest.mark.parametrize("seed", SCHED_SEEDS)
def test_relabel_null_detects_a_planted_within_window_dependence(seed):
    df = _schedule_only_log(seed, plant=SOCIAL)
    res = routine.conditional_mi(df, n_perm=100, rng=np.random.default_rng(seed))
    # Measured under world v2's wider job window over SCHED_SEEDS: 5.792, 6.719, 6.283.
    # The bar moves from 6.0 to 5.0; the plant is still detected on every seed with margin.
    assert res["z"] > 5.0


def test_relabel_null_is_a_bijection_inside_every_schedule_stratum():
    rng = np.random.default_rng(35)
    binned = routine.add_bins(_schedule_only_log(32))
    groups = group_codes(binned, ("agent", "day", "open_code", "night"))
    hb = binned["hour_bin"].to_numpy()
    pair_group, pair_label, row_pair = routine._relabel_pairs(hb, groups)
    new = routine._relabel_within_groups(pair_group, pair_label, row_pair, rng)
    assert (new != hb).any()
    for g in np.unique(groups):
        rows = groups == g
        before = np.unique(hb[rows], return_counts=True)
        after = np.unique(new[rows], return_counts=True)
        assert set(before[0]) == set(after[0])
        assert sorted(before[1]) == sorted(after[1])
        # the same old label always receives the same new label
        assert len(set(zip(hb[rows], new[rows]))) == len(before[0])


# ---------------------------------------------------------------------------
# social
# ---------------------------------------------------------------------------

def _social_log(rng, indicator):
    zone = np.where(indicator, SOCIAL, HOME)
    return make_log(rng, zone=zone, active=np.ones_like(indicator, dtype=bool))


def test_copresence_counts_and_density_under_independence():
    rng = np.random.default_rng(18)
    ind = rng.random((DAYS * T, N)) < 0.3
    df = _social_log(rng, ind)
    counts = social.copresence_counts(df, SOCIAL, N, burn_in_days=BURN_IN)
    assert counts.shape == (N, N)
    assert (np.diag(counts) == 0).all()
    post = ind[BURN_IN * T:]
    assert counts[0, 1] == (post[:, 0] & post[:, 1]).sum()
    null = social.time_shift_null(df, SOCIAL, N, n_perm=200, rng=rng, burn_in_days=BURN_IN)
    assert null.shape == (200, N, N)
    assert social.thresholded_density(counts, null, q=0.95) <= 0.2


def test_thresholded_density_detects_a_constructed_pair():
    rng = np.random.default_rng(19)
    ind = rng.random((DAYS * T, N)) < 0.3
    ind[:, 1] = ind[:, 0]
    df = _social_log(rng, ind)
    counts = social.copresence_counts(df, SOCIAL, N, burn_in_days=BURN_IN)
    null = social.time_shift_null(df, SOCIAL, N, n_perm=200, rng=rng, burn_in_days=BURN_IN)
    adj = social.thresholded_adjacency(counts, null, q=0.95)
    assert adj[0, 1] == 1 and adj[1, 0] == 1
    assert social.thresholded_density(counts, null, q=0.95) >= 2.0 / (N * (N - 1))


def test_observed_over_expected_near_one_under_independence():
    rng = np.random.default_rng(20)
    ind = rng.random((DAYS * T, N)) < 0.3
    df = _social_log(rng, ind)
    oe = social.observed_over_expected(df, SOCIAL, N, burn_in_days=BURN_IN)
    assert np.isnan(np.diag(oe)).all()
    assert 0.9 < social.mean_offdiagonal(oe) < 1.1
    oe_all = social.observed_over_expected(df, None, N, burn_in_days=BURN_IN)
    assert 0.9 < social.mean_offdiagonal(oe_all) < 1.1


def test_arrival_phase_variance_synchronous_and_spread():
    rng = np.random.default_rng(21)
    base = make_log(rng)
    t_day = base["t_day"].to_numpy()
    agent = base["agent"].to_numpy()
    sync = make_log(rng, zone=np.where(t_day >= 100, SOCIAL, HOME))
    assert social.arrival_phase_variance(sync, SOCIAL, ticks_per_day=T, burn_in_days=BURN_IN) == pytest.approx(0.0)
    arrival = agent * (T // N)
    spread = make_log(rng, zone=np.where(t_day >= arrival, SOCIAL, HOME))
    assert social.arrival_phase_variance(spread, SOCIAL, ticks_per_day=T, burn_in_days=BURN_IN) == pytest.approx(1.0, abs=1e-9)


def test_contention_gossip_and_clustering():
    rng = np.random.default_rng(22)
    base = make_log(rng)
    queued = base["t_day"].to_numpy() < 10
    informed = base["agent"].to_numpy() <= base["day"].to_numpy()
    df = make_log(rng, queued=queued, informed=informed)
    assert social.contention_per_agent_day(df, burn_in_days=BURN_IN) == pytest.approx(10.0)
    curve = social.gossip_curve(df)
    assert curve.shape == (DAYS,)
    assert np.allclose(curve, (np.arange(DAYS) + 1) / N)
    triangle = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert social.clustering_coefficient(triangle) == pytest.approx(1.0)
    star = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    assert social.clustering_coefficient(star) == pytest.approx(0.0)
    assert social.density(triangle) == pytest.approx(1.0)
    assert social.density(star) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def test_flags_clean_and_degenerate():
    rng = np.random.default_rng(23)
    clean = make_log(rng)
    res = dashboard.flags(clean, burn_in_days=BURN_IN)
    assert res["degenerate"] is False
    assert res["median_distinct_zones"] == 6
    assert res["mean_entropy_nats"] == pytest.approx(np.log(N_ACTIONS))
    assert res["max_zone_share"] < 0.3

    n = DAYS * T * N
    logp = np.full((n, N_ACTIONS), -np.inf)
    logp[:, HOME] = 0.0
    bad = make_log(rng, zone=np.full(n, HOME), logp=logp)
    res = dashboard.flags(bad, greedy_return=0.0, burn_in_days=BURN_IN)
    assert res["degenerate"] and res["few_zones"] and res["single_zone"] and res["low_entropy"]
    assert res["below_greedy"] is True
    assert res["mean_entropy_nats"] == pytest.approx(0.0)


def test_summary_one_row():
    rng = np.random.default_rng(24)
    df = make_log(rng)
    row = dashboard.summary(df, burn_in_days=BURN_IN)
    assert len(row) == 1
    fracs = row[[f"frac_{z}" for z in ZONE_NAMES]].to_numpy()
    assert fracs.sum() == pytest.approx(1.0)
    assert {"mean_D", "contention_per_agent_day", "gossip_fraction_end", "mean_return", "seed", "condition"} <= set(row.columns)


# ---------------------------------------------------------------------------
# predictive gain: the H1 statistic
# ---------------------------------------------------------------------------

def test_predictive_gain_schedule_only_mean_z_near_zero():
    """Over several seeds the schedule-only process sits at the null on average.

    The criterion is the mean, not only each |z| < 3: a null that occupies
    more of the table than the data reads negative on average.
    """
    rng = np.random.default_rng(5)
    zs = [routine.predictive_gain(routine.add_bins(_schedule_only_log(seed)), n_perm=40, rng=rng)["z"]
          for seed in range(31, 37)]
    assert -0.5 < float(np.mean(zs)) < 1.0, zs
    assert max(abs(z) for z in zs) < 3.5, zs


def test_predictive_gain_detects_planted_dependence():
    rng = np.random.default_rng(6)
    zs = [routine.predictive_gain(routine.add_bins(_schedule_only_log(seed, plant=SOCIAL)), n_perm=40, rng=rng)["z"]
          for seed in (32, 33, 35)]
    assert min(zs) > 4, zs


def test_predictive_gain_is_occupancy_insensitive():
    """Adding noise to the zones (many more occupied cells) cannot create a positive gain."""
    rng = np.random.default_rng(7)
    base = routine.add_bins(_schedule_only_log(34))
    noisy = base.copy()
    flip = rng.random(len(noisy)) < 0.25
    noisy.loc[flip, "zone"] = rng.integers(0, len(ZONE_NAMES), size=int(flip.sum()))
    g_base = routine.predictive_gain(base, n_perm=10, rng=rng)["value"]
    g_noisy = routine.predictive_gain(noisy, n_perm=10, rng=rng)["value"]
    assert g_base <= 0.01 and g_noisy <= 0.01, (g_base, g_noisy)
