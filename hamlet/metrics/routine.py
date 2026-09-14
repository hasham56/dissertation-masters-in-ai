"""Routine metrics: clock information, predictability, periodicity, shock recovery.

All functions take a log DataFrame in ``config.LOG_COLUMNS`` and operate on
the zone sequence of each agent after burn-in. Entropies are in bits, times
are in ticks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from hamlet.config import HamletConfig
from hamlet.metrics.common import (
    DEFAULTS,
    after_burn_in,
    burn_in_or_default,
    default_rng,
    entropy_bits,
    group_codes,
    n_agents_or_infer,
    null_summary,
    tick_agent_matrix,
    ticks_per_day_or_default,
)

_WORLD = HamletConfig()
_STATE_COLUMNS = ("E", "F", "C")
STATE_EDGES = ("quantile", "fixed")
_CLOCK_COLUMNS = ("t_day", "hour_bin", "open_code", "night")


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------

_LAG_PREFIX = "_obs_"
_LAG_KEYS = ("seed", "episode", "agent")


def _lag_state(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``_obs_E``, ``_obs_F``, ``_obs_C``: the state the policy saw when it acted on that tick.

    The log records the state *after* the tick, so the levels on row t are a consequence of the
    action on row t and not an input to it. Binning them makes the state bin partly an echo of the
    behaviour it is meant to condition on, which is what let a uniformly random policy score a state
    gain of 0.044 bits at z 13.59 in the fixed-travel world. The bin is therefore built on the
    previous tick's levels within each agent-episode, which is exactly what the observation carried.

    The shift runs over whatever of ``seed``, ``episode`` and ``agent`` the frame has, ordered by
    ``t``. The first row of each group has no predecessor and keeps its own levels; when the frame
    still holds the burn-in day, as it does wherever this is called before ``after_burn_in``, the
    first metric tick takes the last burn-in tick and no metric row falls back.
    """
    if all(_LAG_PREFIX + c in df.columns for c in _STATE_COLUMNS):
        return df
    out = df.copy()
    keys = [k for k in _LAG_KEYS if k in out.columns]
    cols = list(_STATE_COLUMNS)
    if not keys:
        order = out.sort_values("t").index if "t" in out.columns else out.index
        shifted = out.loc[order, cols].shift(1)
    else:
        order = out.sort_values([*keys, "t"]).index if "t" in out.columns else out.sort_values(keys).index
        shifted = out.loc[order, cols].groupby([out.loc[order, k] for k in keys], sort=False).shift(1)
    shifted = shifted.reindex(out.index)
    for col in cols:
        out[_LAG_PREFIX + col] = shifted[col].fillna(out[col]).astype(np.float64)
    return out


def add_bins(
    df: pd.DataFrame,
    ref: pd.DataFrame | None = None,
    job_open: tuple[int, int] | None = None,
    market_open: tuple[int, int] | None = None,
    hour_bin_ticks: int | None = None,
    night: tuple[int, int] | None = None,
    n_state_bins: int = 3,
    state_edges: str = "quantile",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``hour_bin``, ``state_bin``, ``open_code`` and ``night``.

    ``hour_bin = t_day // hour_bin_ticks`` (8 bins of 3 hours by default).
    ``state_bin`` is the code ``k^2*E_bin + k*F_bin + C_bin`` with
    ``k = n_state_bins`` bins per state (terciles by default, so 27 codes;
    quintiles give 125). With ``state_edges="quantile"`` (the primary
    binning) the bins are equal-frequency and the cut points are taken from
    ``ref`` (default: ``df`` itself; pass the pooled evaluation data of a
    condition to share cut points across runs). With ``state_edges="fixed"``
    (the second pre-declared robustness variant) the cut points are the
    absolute edges ``1/k, ..., (k-1)/k`` of [0, 1], the same for every
    policy, seed and condition, so ``ref`` is ignored.
    ``open_code = 2*job_open + market_open`` as a 4-valued code (both jobs
    share ``job_open``).
    ``night`` is 1 when ``t_day >= night[0] or t_day < night[1]`` (the world
    default is 22:00-06:00), else 0. Together ``open_code`` and ``night`` are
    the schedule: everything a policy can read off the clock without
    counting hours inside a window.
    """
    df = _lag_state(df)
    ref = _lag_state(ref) if ref is not None else df
    wo = job_open if job_open is not None else _WORLD.job_open
    mo = market_open if market_open is not None else _WORLD.market_open
    nt = night if night is not None else _WORLD.night
    width = hour_bin_ticks if hour_bin_ticks is not None else DEFAULTS.hour_bin_ticks

    out = df.copy()
    t_day = out["t_day"].to_numpy()
    out["hour_bin"] = (t_day // width).astype(np.int64)

    k = int(n_state_bins)
    if state_edges not in STATE_EDGES:
        raise ValueError(f"unknown state_edges {state_edges!r}; choose from {STATE_EDGES}")
    code = np.zeros(len(out), dtype=np.int64)
    qs = np.arange(1, k) / k
    for col in _STATE_COLUMNS:
        lagged = _LAG_PREFIX + col
        cuts = qs if state_edges == "fixed" else np.quantile(ref[lagged].to_numpy(dtype=np.float64), qs)
        code = code * k + np.searchsorted(cuts, out[lagged].to_numpy(dtype=np.float64), side="right")
    out["state_bin"] = code

    job_flag = (t_day >= wo[0]) & (t_day < wo[1])
    market_flag = (t_day >= mo[0]) & (t_day < mo[1])
    out["open_code"] = (2 * job_flag + market_flag).astype(np.int64)
    out["night"] = ((t_day >= nt[0]) | (t_day < nt[1])).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# conditional mutual information
# ---------------------------------------------------------------------------

def _plugin_cmi_bits(target: np.ndarray, given: np.ndarray, cond: np.ndarray) -> float:
    """I(target; given | cond) in bits from a count table over dense integer codes.

    Uses H(G,C) + H(T,C) - H(T,G,C) - H(C) with plug-in entropies; the table
    is one ``np.bincount`` on a combined index.
    """
    n_t, n_g, n_c = int(target.max()) + 1, int(given.max()) + 1, int(cond.max()) + 1
    table = np.bincount((cond * n_g + given) * n_t + target, minlength=n_c * n_g * n_t).reshape(n_c, n_g, n_t)
    h_cgt = entropy_bits(table)
    h_cg = entropy_bits(table.sum(axis=2))
    h_ct = entropy_bits(table.sum(axis=1))
    h_c = entropy_bits(table.sum(axis=(1, 2)))
    return h_cg + h_ct - h_cgt - h_c


def _shuffle_within_strata(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute ``values`` independently inside each stratum."""
    n = values.shape[0]
    pos = np.argsort(strata, kind="stable")
    order = np.lexsort((rng.random(n), strata))
    out = np.empty_like(values)
    out[pos] = values[order]
    return out


def _block_shift_index(blocks: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row index that cyclically shifts every block by its own random offset.

    Rows of a block are ordered by ``t``; ``values[out]`` is ``values`` with
    each block's sequence rotated, so every run length inside a block survives
    and only where the runs fall changes.
    """
    n = blocks.shape[0]
    order = np.lexsort((t, blocks))
    b_sorted = blocks[order]
    starts = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1]])
    lengths = np.diff(np.r_[starts, n])
    offsets = rng.integers(0, lengths)
    block_rank = np.repeat(np.arange(starts.size), lengths)
    pos = np.arange(n) - starts[block_rank]
    src_sorted = starts[block_rank] + (pos + offsets[block_rank]) % lengths[block_rank]
    src = np.empty(n, dtype=np.int64)
    src[order] = order[src_sorted]
    return src


def _relabel_pairs(labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct (group, label) pairs, sorted by group, and each row's pair index.

    Computed once per call of :func:`conditional_mi`; every null draw then only
    permutes the pair table inside each group.
    """
    n_l = int(labels.max()) + 1
    uniq, inv = np.unique(groups * n_l + labels, return_inverse=True)
    return uniq // n_l, uniq % n_l, inv.reshape(-1)


def _relabel_within_groups(
    pair_group: np.ndarray, pair_label: np.ndarray, row_pair: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Apply a random bijection to the labels present in each group.

    ``pair_group`` is non-decreasing, so the pairs of one group are a
    contiguous block; sorting the block by a random key gives a permutation
    of the labels that occur in that group and of no others. Rows that share
    a label inside a group receive the same new label, so every run and every
    cell count survives and only the names of the labels change.
    """
    order = np.lexsort((rng.random(pair_group.size), pair_group))
    return pair_label[order][row_pair]


def conditional_mi(
    df: pd.DataFrame,
    target: str = "zone",
    given: tuple[str, ...] = ("hour_bin",),
    cond: tuple[str, ...] = ("state_bin", "open_code", "night"),
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    burn_in_days: int | None = None,
    null: str = "relabel",
    relabel_scope: str = "agent_day",
    report_cells: bool = False,
) -> dict[str, float]:
    """Plug-in conditional mutual information I(target; given | cond) in bits.

    ``relabel_scope`` sets the unit that receives one bijection under the
    relabel null: ``"agent_day"`` (each agent-day and stratum separately),
    ``"agent"`` (one bijection per agent and stratum, shared across its days)
    or ``"day"`` (one per day and stratum, shared across agents). With
    ``report_cells`` the result also carries the number of occupied
    (cond, given, target) and (cond, given) cells in the data and, on
    average, under the null, which is the check that the null does not
    occupy more of the table than the data does.

    With the defaults this is the clock information beyond the schedule,
    I(zone; hour_bin | state_bin, open_code, night): does the hour inside a
    window predict the zone once the state terciles and everything the clock
    says about the schedule (which zones are open, whether it is night) are
    known. GREEDY-CLOCK reads exactly the schedule and nothing else, so it is
    the zero-beyond-schedule reference for this statistic; a learned policy
    that anticipates (eats before the canteen fills, rests before night) is
    what would score above it.

    Three nulls are available; all leave the target untouched.

    ``"relabel"`` (default)
        Inside each (agent, day, schedule stratum) group, where the schedule
        strata are the clock columns of ``cond`` (``open_code`` and ``night``
        by default), draw a random bijection of the ``given`` labels that
        occur in that group and rename the rows accordingly. Every run, every
        stratum, every (state, hour) cell count within a stratum and the lock
        between the schedule and the trajectory survive; what is destroyed is
        which hour inside a window an activity falls at, which is exactly the
        information the statistic is about. On evaluation seeds 10000-10002
        RANDOM scores z between -1.4 and -0.1, GREEDY-STATE between -0.3 and
        +4.3, GREEDY-CLOCK between +1.4 and +3.9, and a planted within-window
        dependence (zone set to SOCIAL with probability 0.8 from 12:00 to
        15:00 inside the work window) scores +8 to +11 on GREEDY-CLOCK and
        more on the other two. The residual of +4 on one seed of each greedy
        policy is the phase persistence of a deterministic need cycle that
        the state terciles do not fully capture, not a bias of the null. The
        null does assume that the hour is not locked to the state terciles
        inside a window beyond what the schedule imposes; a stochastic
        schedule-only process with a slow integrator and a global phase
        breaks that and scores negative, see the synthetic tests.

    ``"block"``
        Cyclically shift the ``given`` labels of every agent-day by a random
        offset along that day. Columns of ``cond`` that are functions of the
        clock (``t_day``, ``hour_bin``, ``open_code``, ``night``) move with
        the ``given`` columns whenever ``given`` is itself a clock column, so
        the whole clock is shifted by a random number of ticks each day and
        the alignment of the trajectory with the clock is destroyed. This is
        the right null for a policy that ignores the clock: RANDOM and
        GREEDY-STATE score |z| below 1.5. It is biased for any policy that
        follows the schedule. In the data the real hour is locked to the
        schedule and hence to the state trajectory (H(hour_bin | state_bin)
        is about 1.7 bits for GREEDY-CLOCK against 3.0 for RANDOM); the
        shifted hour is not, so the null occupies more (stratum, hour) cells
        than the data and carries more plug-in bias, and GREEDY-CLOCK scores
        z between -5.3 and -2.8 while using no within-window information at
        all. Bias
        correction of the estimator and held-out predictive gain do not cure
        this, because the effective sample size under autocorrelation is
        unknown; they flip the sign instead. Kept as the clock-blind
        reference null and for the state-MI.

    ``"tick"``
        Shuffle the ``given`` labels among the rows of each ``cond`` stratum.
        It keeps every marginal and the conditioning structure but treats
        consecutive ticks as exchangeable, which they are not: zones change
        one tile per tick and an hour bin lasts 30 ticks, so the null is far
        too narrow and even a random walker scores a large z. Kept for
        comparison only.

    Do not add ``prev_zone`` to the conditioning set: it reintroduces the
    support bias of the block null (GREEDY-CLOCK back at z -4.0 to -4.7
    under the relabel null, the clock-blind policies unchanged).

    The plug-in estimate is biased upwards by roughly (cells / 2n ln 2) bits;
    the null carries the same bias, so ``z`` and ``value - null_mean`` are the
    quantities to report. Missing bin columns are added with :func:`add_bins`
    using ``df`` itself as the tercile reference.
    Returns ``{"value", "null_mean", "null_sd", "z"}``.
    """
    rng = default_rng(rng)
    n_perm = DEFAULTS.n_perm if n_perm is None else int(n_perm)
    data = after_burn_in(df, burn_in_days)
    needed = set(given) | set(cond) | {target}
    if not needed.issubset(data.columns):
        data = add_bins(data)
    t = group_codes(data, target)
    g = group_codes(data, given)
    c = group_codes(data, cond) if len(cond) else np.zeros(len(data), dtype=np.int64)
    value = _plugin_cmi_bits(t, g, c)
    if null == "tick":
        null_values = np.array([_plugin_cmi_bits(t, _shuffle_within_strata(g, c, rng), c) for _ in range(n_perm)])
    elif null == "block":
        clock_given = any(col in _CLOCK_COLUMNS for col in given)
        moving = [col for col in cond if clock_given and col in _CLOCK_COLUMNS]
        fixed = [col for col in cond if col not in moving]
        c_move = group_codes(data, moving) if moving else np.zeros(len(data), dtype=np.int64)
        c_fix = group_codes(data, fixed) if fixed else np.zeros(len(data), dtype=np.int64)
        n_move = int(c_move.max()) + 1 if c_move.size else 1
        blocks = group_codes(data, ("agent", "day"))
        ticks = data["t"].to_numpy()
        null_values = np.empty(n_perm)
        for i in range(n_perm):
            src = _block_shift_index(blocks, ticks, rng)
            null_values[i] = _plugin_cmi_bits(t, g[src], c_fix * n_move + c_move[src])
    elif null == "relabel":
        strata = [col for col in cond if col in _CLOCK_COLUMNS and col not in given]
        unit = {"agent_day": ("agent", "day"), "agent": ("agent",), "day": ("day",)}[relabel_scope]
        groups = group_codes(data, (*unit, *strata))
        pair_group, pair_label, row_pair = _relabel_pairs(g, groups)
        null_values = np.empty(n_perm)
        cells_null = np.empty((n_perm, 2))
        for i in range(n_perm):
            g_null = _relabel_within_groups(pair_group, pair_label, row_pair, rng)
            null_values[i] = _plugin_cmi_bits(t, g_null, c)
            if report_cells:
                cells_null[i] = _occupied_cells(t, g_null, c)
    else:
        raise ValueError(f"unknown null {null!r}; use 'relabel', 'block' or 'tick'")
    out = null_summary(value, null_values)
    if report_cells and null == "relabel":
        real = _occupied_cells(t, g, c)
        out.update(cells_cgt_real=real[0], cells_cg_real=real[1],
                   cells_cgt_null=float(cells_null[:, 0].mean()), cells_cg_null=float(cells_null[:, 1].mean()))
    return out


def _occupied_cells(target: np.ndarray, given: np.ndarray, cond: np.ndarray) -> tuple[float, float]:
    """Number of occupied (cond, given, target) and (cond, given) cells."""
    n_t, n_g = int(target.max()) + 1, int(given.max()) + 1
    cgt = np.unique((cond * n_g + given) * n_t + target).size
    cg = np.unique(cond * n_g + given).size
    return float(cgt), float(cg)


def _shrunk_gain(
    target: np.ndarray, given: np.ndarray, cond: np.ndarray, day: np.ndarray, n_target: int,
    pseudo_count: float, alpha: float,
) -> float:
    """Held-out log2-likelihood gain per row from adding ``given`` to ``cond``.

    Two folds over alternating days. P(target | cond) is estimated with
    add-``alpha`` smoothing; P(target | cond, given) is shrunk towards it with
    ``pseudo_count`` pseudo-observations, so a (cond, given) cell seen a few
    times contributes almost nothing and an unseen one contributes exactly
    nothing. The estimate therefore does not grow with the number of occupied
    cells, which is what makes it comparable across policies of different
    stochasticity and between the data and a permutation null.
    """
    days = np.unique(day)
    folds = [(np.isin(day, days[::2]), np.isin(day, days[1::2]))]
    folds.append((folds[0][1], folds[0][0]))
    n_g = int(given.max()) + 1
    n_c = int(cond.max()) + 1
    cg = cond * n_g + given
    gains = []
    for tr, te in folds:
        base = np.bincount(cond[tr] * n_target + target[tr], minlength=n_c * n_target).reshape(n_c, n_target).astype(float)
        p_base = (base + alpha) / (base + alpha).sum(axis=1, keepdims=True)
        full = np.bincount(cg[tr] * n_target + target[tr], minlength=n_c * n_g * n_target).reshape(n_c * n_g, n_target).astype(float)
        prior = np.repeat(p_base, n_g, axis=0)
        p_full = (full + pseudo_count * prior) / (full.sum(axis=1, keepdims=True) + pseudo_count)
        gains.append(np.log2(p_full[cg[te], target[te]]).mean() - np.log2(p_base[cond[te], target[te]]).mean())
    return float(np.mean(gains))


def predictive_gain(
    df: pd.DataFrame,
    target: str = "zone",
    given: tuple[str, ...] = ("hour_bin",),
    cond: tuple[str, ...] = ("state_bin", "open_code", "night"),
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    burn_in_days: int | None = None,
    pseudo_count: float = 50.0,
    alpha: float = 0.5,
) -> dict[str, float]:
    """Clock information beyond the schedule as a held-out predictive gain, in bits per tick.

    This is the H1 statistic. It asks how much better the zone is predicted on
    held-out days when the hour bin is added to the state bins and the
    schedule (opening hours, night), with the finer model shrunk towards the
    coarser one (see :func:`_shrunk_gain`). Unlike the plug-in conditional
    mutual information it carries no positive bias from occupied cells, so a
    stochastic learned policy and a deterministic scheduler can be compared
    on it directly; GREEDY-CLOCK, which reads the schedule and nothing else,
    is the zero-beyond-schedule reference and H1 is the seed-paired contrast
    of a learned policy against it.

    The null relabels hour bins within each (agent, day, schedule stratum)
    group, as in :func:`conditional_mi`. It is validated on clock-blind
    policies and on a synthetic schedule-only process (mean z within 0.5 of
    zero over twelve seeds); for processes with very slow state integration it
    is conservative (mean z about -1), so the null z is a sanity check and not
    the primary test. Returns ``{"value", "null_mean", "null_sd", "z"}``.
    """
    rng = default_rng(rng)
    n_perm = DEFAULTS.n_perm if n_perm is None else int(n_perm)
    data = after_burn_in(df, burn_in_days)
    needed = set(given) | set(cond) | {target}
    if not needed.issubset(data.columns):
        data = add_bins(data)
    t = group_codes(data, target)
    g = group_codes(data, given)
    c = group_codes(data, cond) if len(cond) else np.zeros(len(data), dtype=np.int64)
    day = data["day"].to_numpy()
    n_target = int(t.max()) + 1
    value = _shrunk_gain(t, g, c, day, n_target, pseudo_count, alpha)
    strata = [col for col in cond if col in _CLOCK_COLUMNS and col not in given]
    groups = group_codes(data, ("agent", "day", *strata))
    pair_group, pair_label, row_pair = _relabel_pairs(g, groups)
    null_values = np.array([
        _shrunk_gain(t, _relabel_within_groups(pair_group, pair_label, row_pair, rng), c, day, n_target, pseudo_count, alpha)
        for _ in range(n_perm)
    ])
    return null_summary(value, null_values)


# ---------------------------------------------------------------------------
# entropy rate and predictability (Song et al. 2010)
# ---------------------------------------------------------------------------

def _match_lengths(seq: np.ndarray) -> np.ndarray:
    """Lambda_i: shortest substring starting at i not seen starting before i.

    Matches may overlap the current position, as in Kontoyiannis et al. (1998).
    Symbols are mapped to bytes so the search runs in C. If no unseen substring
    exists before the end of the sequence, Lambda_i = n - i + 1.
    """
    symbols, codes = np.unique(seq, return_inverse=True)
    if symbols.size > 255:
        raise ValueError("at most 255 distinct symbols are supported")
    buf = codes.astype(np.uint8).tobytes()
    n = len(buf)
    lam = np.empty(n, dtype=np.int64)
    for i in range(n):
        length = 1
        while i + length <= n:
            if buf.find(buf[i:i + length], 0, i + length - 1) < 0:
                break
            length += 1
        lam[i] = length if i + length <= n else n - i + 1
    return lam


def lz_entropy_rate(seq: np.ndarray) -> float:
    """Lempel-Ziv entropy-rate estimate in bits per symbol.

    S = (n / sum_i Lambda_i) * log2(n), the estimator of Kontoyiannis et al.
    (1998) as used by Song et al. (2010), where Lambda_i is the length of the
    shortest substring starting at position i that has not appeared before.
    """
    seq = np.asarray(seq)
    n = seq.shape[0]
    if n < 2:
        return 0.0
    lam = _match_lengths(seq)
    return float(n / lam.sum() * np.log2(n))


def random_entropy(seq: np.ndarray) -> float:
    """S_rand = log2 of the number of distinct symbols (bits)."""
    return float(np.log2(np.unique(np.asarray(seq)).size))


def uncorrelated_entropy(seq: np.ndarray) -> float:
    """S_unc: Shannon entropy of the symbol frequencies, order ignored (bits)."""
    _, counts = np.unique(np.asarray(seq), return_counts=True)
    return entropy_bits(counts)


def fano_predictability(S: float, n_symbols: int) -> float:
    """Maximum predictability Pi_max from Fano's inequality (Song et al. 2010).

    Solves S = H(Pi) + (1 - Pi) log2(N - 1) for Pi in [1/N, 1], where
    H is the binary entropy in bits and N is the number of symbols.
    """
    N = int(n_symbols)
    if N <= 1 or S <= 0:
        return 1.0
    if S >= np.log2(N):
        return 1.0 / N

    def f(pi: float) -> float:
        h = -pi * np.log2(pi) - (1 - pi) * np.log2(1 - pi) if 0 < pi < 1 else 0.0
        return h + (1 - pi) * np.log2(N - 1) - S

    return float(brentq(f, 1.0 / N, 1.0 - 1e-12))


def predictability_summary(seq: np.ndarray) -> dict[str, float]:
    """S, S_unc, S_rand and their Fano predictability ceilings for one sequence."""
    n_symbols = int(np.unique(np.asarray(seq)).size)
    s = lz_entropy_rate(seq)
    s_unc = uncorrelated_entropy(seq)
    s_rand = random_entropy(seq)
    return {
        "S": s, "S_unc": s_unc, "S_rand": s_rand,
        "Pi_max": fano_predictability(s, n_symbols),
        "Pi_unc": fano_predictability(s_unc, n_symbols),
        "Pi_rand": fano_predictability(s_rand, n_symbols),
    }


def agent_sequences(df: pd.DataFrame, burn_in_days: int | None = None, n_agents: int | None = None) -> list[np.ndarray]:
    """Zone sequence per agent after burn-in, in tick order."""
    data = after_burn_in(df, burn_in_days)
    n = n_agents_or_infer(data, n_agents)
    a = data["agent"].to_numpy()
    z = data["zone"].to_numpy()
    return [z[a == k] for k in range(n)]


# ---------------------------------------------------------------------------
# regularity and periodicity
# ---------------------------------------------------------------------------

def regularity(
    df: pd.DataFrame,
    ticks_per_day: int | None = None,
    slot: int | None = None,
    burn_in_days: int | None = None,
    n_zones: int | None = None,
    per_agent: bool = False,
) -> float | np.ndarray:
    """Regularity R = mean over clock slots of max_j p(zone = j | slot).

    Computed per agent over the days after burn-in (Song et al. 2010); the
    population value is the mean over agents. A lower bound on predictability
    that ignores order within a slot.
    """
    T = ticks_per_day_or_default(ticks_per_day)
    width = slot if slot is not None else DEFAULTS.hour_bin_ticks
    data = after_burn_in(df, burn_in_days)
    n = n_agents_or_infer(data, None)
    n_slots = int(np.ceil(T / width))
    Z = n_zones if n_zones is not None else int(data["zone"].max()) + 1
    a = data["agent"].to_numpy().astype(np.int64)
    s = (data["t_day"].to_numpy() // width).astype(np.int64)
    z = data["zone"].to_numpy().astype(np.int64)
    counts = np.bincount((a * n_slots + s) * Z + z, minlength=n * n_slots * Z).reshape(n, n_slots, Z)
    tot = counts.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_max = np.where(tot > 0, counts.max(axis=2) / np.where(tot > 0, tot, 1), np.nan)
    r = np.nanmean(p_max, axis=1)
    return r if per_agent else float(np.nanmean(r))


def periodogram_power_at_day(
    df: pd.DataFrame,
    zone: int,
    ticks_per_day: int | None = None,
    burn_in_days: int | None = None,
    per_agent: bool = False,
) -> float | np.ndarray:
    """Fraction of periodogram power within one bin of the daily frequency.

    For each agent the one-hot indicator of ``zone`` over the post-burn-in
    ticks is mean-centred and its periodogram taken; the value is the power in
    bins ``k - 1 .. k + 1`` (k = number of days) divided by the power in all
    non-zero-frequency bins. Agents that never (or always) occupy the zone
    have undefined power and are ignored in the mean.
    """
    T = ticks_per_day_or_default(ticks_per_day)
    data = after_burn_in(df, burn_in_days)
    ind = tick_agent_matrix(data, (data["zone"].to_numpy() == zone).astype(np.float64))
    n_ticks = ind.shape[0]
    k = int(round(n_ticks / T))
    x = ind - ind.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2
    lo, hi = max(k - 1, 1), min(k + 1, power.shape[0] - 1)
    total = power[1:].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, power[lo:hi + 1].sum(axis=0) / np.where(total > 0, total, 1), np.nan)
    return frac if per_agent else float(np.nanmean(frac)) if np.isfinite(frac).any() else float("nan")


def within_day_shuffle(df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Surrogate log with each agent's zone labels permuted within each day.

    Keeps every agent-day's zone composition and destroys within-day timing,
    which removes clock structure while preserving time budgets.
    """
    rng = default_rng(rng)
    out = df.copy()
    a = out["agent"].to_numpy()
    d = out["day"].to_numpy()
    t = out["t"].to_numpy()
    z = out["zone"].to_numpy()
    by_time = np.lexsort((t, d, a))
    by_random = np.lexsort((rng.random(len(out)), d, a))
    new_zone = np.empty_like(z)
    new_zone[by_time] = z[by_random]
    out["zone"] = new_zone
    return out


# ---------------------------------------------------------------------------
# actogram and shock recovery
# ---------------------------------------------------------------------------

def actogram_matrix(df: pd.DataFrame, agent: int, ticks_per_day: int | None = None) -> np.ndarray:
    """Zone id per (day, t_day) for one agent; shape (n_days, ticks_per_day).

    All days are included (burn-in too) since the actogram is a figure, not a
    statistic. Missing entries are -1.
    """
    T = ticks_per_day_or_default(ticks_per_day)
    sub = df[df["agent"].to_numpy() == agent]
    n_days = int(sub["day"].max()) + 1 if len(sub) else 0
    out = np.full((n_days, T), -1, dtype=np.int64)
    out[sub["day"].to_numpy(), sub["t_day"].to_numpy()] = sub["zone"].to_numpy()
    return out


def _day_zone_matrix(df: pd.DataFrame, ticks_per_day: int) -> np.ndarray:
    """Zone id per (agent, day, t_day); shape (N, n_days, ticks_per_day), -1 if missing."""
    n = n_agents_or_infer(df, None)
    n_days = int(df["day"].max()) + 1
    out = np.full((n, n_days, ticks_per_day), -1, dtype=np.int64)
    out[df["agent"].to_numpy(), df["day"].to_numpy(), df["t_day"].to_numpy()] = df["zone"].to_numpy()
    return out


def day_jaccard(df: pd.DataFrame, day_a: int, day_b: int, ticks_per_day: int | None = None) -> float:
    """Mean over agents of the Jaccard similarity between two days' zone sequences.

    Each day is the set of (t_day, zone) pairs; with one zone per tick the
    similarity is m / (2T - m) where m is the number of ticks with equal zone.
    """
    T = ticks_per_day_or_default(ticks_per_day)
    m = _day_zone_matrix(df, T)
    if day_a >= m.shape[1] or day_b >= m.shape[1] or day_a < 0 or day_b < 0:
        return float("nan")
    matches = (m[:, day_a, :] == m[:, day_b, :]).sum(axis=1)
    return float(np.mean(matches / (2 * T - matches)))


def _consecutive_jaccard(df: pd.DataFrame, first_day: int, last_day: int, T: int) -> float:
    pairs = [day_jaccard(df, d, d + 1, T) for d in range(first_day, last_day)]
    return float(np.mean(pairs)) if pairs else float("nan")


def shock_recovery(
    df_shocked: pd.DataFrame,
    df_control: pd.DataFrame,
    shock_day: int | None = None,
    ticks_per_day: int | None = None,
    tol: float | None = None,
    burn_in_days: int | None = None,
) -> dict[str, float]:
    """Drive gap, recovery time and schedule re-formation after a shock.

    ``drive_gap_post`` (the H1c primary): mean population drive D over the
    day after the shock day in the shocked run minus the same in the paired
    control; ``drive_gap_shock`` the same on the shock day itself.
    ``recovery_ticks``: ticks after the end of the shock day until the trailing
    one-day mean of population drive D in the shocked run is within ``tol``
    (relative) of the control's trailing mean at the same tick; NaN if it never
    returns within the episode. ``jaccard_pre`` is the mean consecutive-day
    Jaccard similarity of zone sequences before the shock (after burn-in),
    ``jaccard_shock`` compares the day before the shock with the shock day, and
    ``jaccard_post`` compares the day before the shock with the day after it.
    The ``_control`` entries give the same quantities for the control run.
    ``jaccard_ref_control`` is the control's similarity between the two days
    before the shock (shock_day - 2 against shock_day - 1) and
    ``reformation = jaccard_post - jaccard_ref_control``: how much less the
    day after the shock resembles the day before it than two ordinary
    consecutive days resemble each other in the control, 0 when the schedule
    re-forms exactly and negative when it does not (the plan reads
    re-formation as ``reformation >= -0.1``).
    """
    T = ticks_per_day_or_default(ticks_per_day)
    sd = shock_day if shock_day is not None else _WORLD.shock_day
    tol = tol if tol is not None else DEFAULTS.recovery_tol
    b = burn_in_or_default(burn_in_days)

    ds = df_shocked.groupby("t")["D"].mean()
    dc = df_control.groupby("t")["D"].mean()
    common = ds.index.intersection(dc.index).sort_values()
    ds_v = ds.loc[common].to_numpy(dtype=np.float64)
    dc_v = dc.loc[common].to_numpy(dtype=np.float64)
    t = common.to_numpy()
    recovery = float("nan")
    if len(t) >= T:
        kernel = np.ones(T) / T
        ms = np.convolve(ds_v, kernel, mode="valid")
        mc = np.convolve(dc_v, kernel, mode="valid")
        t_end = t[T - 1:]
        start = (sd + 1) * T
        ok = (t_end >= start) & (np.abs(ms - mc) <= tol * np.abs(mc))
        hits = np.flatnonzero(ok)
        if hits.size:
            recovery = float(t_end[hits[0]] - start)

    def day_drive(frame: pd.DataFrame, d: int) -> float:
        sel = frame.loc[frame["day"] == d, "D"]
        return float(sel.mean()) if len(sel) else float("nan")

    out = {
        "drive_gap_post": day_drive(df_shocked, sd + 1) - day_drive(df_control, sd + 1),
        "drive_gap_shock": day_drive(df_shocked, sd) - day_drive(df_control, sd),
        "recovery_ticks": recovery,
    }
    for name, frame in (("", df_shocked), ("_control", df_control)):
        out["jaccard_pre" + name] = _consecutive_jaccard(frame, b, sd - 1, T)
        out["jaccard_shock" + name] = day_jaccard(frame, sd - 1, sd, T)
        out["jaccard_post" + name] = day_jaccard(frame, sd - 1, sd + 1, T)
    out["jaccard_ref_control"] = day_jaccard(df_control, sd - 2, sd - 1, T)
    out["reformation"] = out["jaccard_post"] - out["jaccard_ref_control"]
    return out
