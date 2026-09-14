"""Social metrics: co-presence networks, association, synchrony, contention, gossip.

Co-presence means two agents both hold an active slot in the same zone on the
same tick. Everything is computed on days after burn-in except the gossip
curve, which tracks the whole episode. Counts are in ticks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hamlet.config import SOCIAL
from hamlet.metrics.common import (
    DEFAULTS,
    after_burn_in,
    default_rng,
    n_agents_or_infer,
    tick_agent_matrix,
    ticks_per_day_or_default,
)


def _active_at(df: pd.DataFrame, zone: int, n_agents: int | None) -> np.ndarray:
    """(T, N) 0/1 matrix: agent active at ``zone`` on each tick of ``df``."""
    ind = (df["zone"].to_numpy() == zone) & df["active"].to_numpy().astype(bool)
    return tick_agent_matrix(df, ind.astype(np.int64), n_agents)


def _counts_from_indicator(ind: np.ndarray) -> np.ndarray:
    counts = ind.T @ ind
    np.fill_diagonal(counts, 0)
    return counts


def copresence_counts(
    df: pd.DataFrame,
    zone: int = SOCIAL,
    n_agents: int | None = None,
    burn_in_days: int | None = None,
) -> np.ndarray:
    """(N, N) ticks on which both agents were active at ``zone``; zero diagonal."""
    data = after_burn_in(df, burn_in_days)
    return _counts_from_indicator(_active_at(data, zone, n_agents))


def time_shift_null(
    df: pd.DataFrame,
    zone: int = SOCIAL,
    n_agents: int | None = None,
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    burn_in_days: int | None = None,
) -> np.ndarray:
    """(n_perm, N, N) co-presence counts under independent cyclic time shifts.

    Each agent's active-at-zone indicator is rotated by its own random offset
    drawn uniformly from ``0 .. T - 1``, which keeps every agent's temporal
    structure and destroys the pairing between agents.
    """
    rng = default_rng(rng)
    n_perm = DEFAULTS.n_perm if n_perm is None else int(n_perm)
    data = after_burn_in(df, burn_in_days)
    ind = _active_at(data, zone, n_agents)
    T, N = ind.shape
    rows = np.arange(T)[:, None]
    cols = np.arange(N)[None, :]
    out = np.empty((n_perm, N, N), dtype=np.int64)
    for p in range(n_perm):
        offsets = rng.integers(0, T, size=N)
        shifted = ind[(rows - offsets[None, :]) % T, cols]
        out[p] = _counts_from_indicator(shifted)
    return out


def thresholded_adjacency(counts: np.ndarray, null: np.ndarray, q: float | None = None) -> np.ndarray:
    """Symmetric 0/1 adjacency: edge iff counts exceed the ``q`` quantile of the null per pair."""
    q = DEFAULTS.density_quantile if q is None else float(q)
    thr = np.quantile(np.asarray(null, dtype=np.float64), q, axis=0)
    adj = (np.asarray(counts) > thr).astype(np.int64)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    return adj


def density(adjacency: np.ndarray) -> float:
    """2|E| / (N (N - 1)) of a symmetric 0/1 adjacency matrix."""
    a = np.asarray(adjacency)
    n = a.shape[0]
    if n < 2:
        return float("nan")
    return float(np.triu(a, k=1).sum() * 2.0 / (n * (n - 1)))


def thresholded_density(counts: np.ndarray, null: np.ndarray, q: float | None = None) -> float:
    """Network density after thresholding co-presence counts against the time-shift null."""
    return density(thresholded_adjacency(counts, null, q))


def observed_over_expected(
    df: pd.DataFrame,
    zone: int | None = SOCIAL,
    n_agents: int | None = None,
    burn_in_days: int | None = None,
) -> np.ndarray:
    """(N, N) observed-over-expected co-presence; NaN on the diagonal.

    O_ij is the number of ticks both are active at the zone and
    E_ij = T * sum_l p_i(l) p_j(l) where p_i(l) is agent i's fraction of ticks
    active at zone l and the sum runs over the chosen zone only (``zone`` an
    int) or over every zone (``zone=None``), the usual animal-association
    statistic. Values above 1 indicate attraction, below 1 avoidance.
    """
    data = after_burn_in(df, burn_in_days)
    zones = sorted(int(z) for z in np.unique(data["zone"].to_numpy())) if zone is None else [int(zone)]
    n = n_agents_or_infer(data, n_agents)
    observed = np.zeros((n, n), dtype=np.float64)
    expected = np.zeros((n, n), dtype=np.float64)
    T = 0
    for z in zones:
        ind = _active_at(data, z, n).astype(np.float64)
        T = ind.shape[0]
        observed += ind.T @ ind
        p = ind.mean(axis=0)
        expected += np.outer(p, p)
    expected *= T
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(expected > 0, observed / np.where(expected > 0, expected, 1.0), np.nan)
    np.fill_diagonal(ratio, np.nan)
    return ratio


def mean_offdiagonal(matrix: np.ndarray) -> float:
    """Mean of the strictly upper triangle, ignoring NaN."""
    m = np.asarray(matrix, dtype=np.float64)
    iu = np.triu_indices(m.shape[0], k=1)
    vals = m[iu]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def arrival_phase_variance(
    df: pd.DataFrame,
    zone: int = SOCIAL,
    ticks_per_day: int | None = None,
    burn_in_days: int | None = None,
    pooled: bool = False,
) -> float:
    """Circular variance of the t_day at which agents first become active at ``zone`` each day.

    Phase is 2*pi*t_day/ticks_per_day. By default the variance is taken
    across agents within each day and averaged over days (synchrony of
    arrival); with ``pooled`` every agent-day arrival enters one distribution.
    Days with fewer than two arrivals are ignored. 0 means everyone arrives at
    the same clock time, 1 means arrivals are spread around the clock.
    """
    T = ticks_per_day_or_default(ticks_per_day)
    data = after_burn_in(df, burn_in_days)
    mask = (data["zone"].to_numpy() == zone) & data["active"].to_numpy().astype(bool)
    hits = data[mask]
    first = hits.groupby(["day", "agent"])["t_day"].min()
    if first.empty:
        return float("nan")
    phase = np.exp(1j * 2 * np.pi * first.to_numpy(dtype=np.float64) / T)
    if pooled:
        return float(1.0 - np.abs(phase.mean())) if phase.size > 1 else float("nan")
    days = first.index.get_level_values("day").to_numpy()
    out = []
    for d in np.unique(days):
        ph = phase[days == d]
        if ph.size > 1:
            out.append(1.0 - np.abs(ph.mean()))
    return float(np.mean(out)) if out else float("nan")


def contention_per_agent_day(df: pd.DataFrame, burn_in_days: int | None = None) -> float:
    """Mean number of refused (queued) ticks per agent per day after burn-in."""
    data = after_burn_in(df, burn_in_days)
    n_days = int(data["day"].nunique())
    n = n_agents_or_infer(data, None)
    if n_days == 0:
        return float("nan")
    return float(data["queued"].to_numpy().astype(bool).sum() / (n * n_days))


def gossip_curve(df: pd.DataFrame) -> np.ndarray:
    """Fraction of agents informed at the last tick of each day, over all days."""
    last_tick = df.groupby("day")["t"].max()
    sub = df[df["t"].isin(last_tick.to_numpy())]
    frac = sub.groupby("day")["informed"].mean().sort_index()
    return frac.to_numpy(dtype=np.float64)


def clustering_coefficient(adjacency: np.ndarray) -> float:
    """Global clustering coefficient: 3 * triangles / connected triples via trace(A^3)."""
    a = np.asarray(adjacency, dtype=np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    triples = (deg * (deg - 1)).sum()
    if triples <= 0:
        return float("nan")
    triangles = np.trace(a @ a @ a)
    return float(triangles / triples)


def density_over_days(
    df: pd.DataFrame,
    zone: int = SOCIAL,
    n_agents: int | None = None,
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    q: float | None = None,
) -> np.ndarray:
    """Thresholded density for every one-day window of the episode (no burn-in filter)."""
    days = np.unique(df["day"].to_numpy())
    out = np.empty(days.size, dtype=np.float64)
    for i, d in enumerate(days):
        sub = df[df["day"].to_numpy() == d]
        counts = copresence_counts(sub, zone, n_agents, burn_in_days=0)
        null = time_shift_null(sub, zone, n_agents, n_perm, rng, burn_in_days=0)
        out[i] = thresholded_density(counts, null, q)
    return out
