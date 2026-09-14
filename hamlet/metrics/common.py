"""Helpers shared by the metric modules.

Every metric takes a DataFrame in ``config.LOG_COLUMNS`` for one evaluation
episode (or a pooled set of episodes) and returns plain floats or small dicts.
The helpers here handle burn-in filtering, scattering a per-row vector into a
(tick, agent) matrix, plug-in entropies in bits and the summary of a
permutation null.

Analysis constants that are not part of the world (bin widths, permutation
counts, thresholds) live in :class:`MetricsDefaults`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from hamlet.config import HamletConfig

_WORLD = HamletConfig()


@dataclass(frozen=True)
class MetricsDefaults:
    """Analysis constants used by the metric functions.

    Attributes:
        hour_bin_ticks: width of one hour bin in ticks (30 ticks = 3 hours).
        n_perm: permutations drawn for every null distribution.
        n_boot: bootstrap resamples for confidence intervals.
        alpha: two-sided error rate for intervals and tests.
        density_quantile: null quantile above which a co-presence edge exists.
        recovery_tol: relative tolerance on daily mean drive for shock recovery.
        min_zones_per_day: degeneracy flag if the median number of distinct
            zones visited per agent-day falls below this.
        max_zone_share: degeneracy flag if any agent spends more than this
            fraction of its ticks in one zone.
        min_entropy_nats: degeneracy flag if mean policy entropy is below this.
    """

    hour_bin_ticks: int = 30
    n_perm: int = 200
    n_boot: int = 2000
    alpha: float = 0.05
    density_quantile: float = 0.95
    recovery_tol: float = 0.10
    min_zones_per_day: int = 3
    max_zone_share: float = 0.70
    min_entropy_nats: float = 0.05


DEFAULTS = MetricsDefaults()


def default_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return ``rng`` or a fresh generator seeded from the default source."""
    return rng if rng is not None else np.random.default_rng()


def ticks_per_day_or_default(ticks_per_day: int | None) -> int:
    """Episode clock length in ticks; falls back to the world default."""
    return int(ticks_per_day) if ticks_per_day is not None else _WORLD.ticks_per_day


def burn_in_or_default(burn_in_days: int | None) -> int:
    """Number of leading days excluded from every metric."""
    return int(burn_in_days) if burn_in_days is not None else _WORLD.burn_in_days


def after_burn_in(df: pd.DataFrame, burn_in_days: int | None = None) -> pd.DataFrame:
    """Rows with ``day >= burn_in_days``, sorted by tick then agent."""
    b = burn_in_or_default(burn_in_days)
    out = df[df["day"].to_numpy() >= b]
    return out.sort_values(["t", "agent"], kind="stable").reset_index(drop=True)


def n_agents_or_infer(df: pd.DataFrame, n_agents: int | None) -> int:
    """Population size, inferred from the largest agent id when not given."""
    return int(n_agents) if n_agents is not None else int(df["agent"].max()) + 1


def tick_agent_matrix(df: pd.DataFrame, values: np.ndarray, n_agents: int | None = None) -> np.ndarray:
    """Scatter a per-row vector into a (T, N) array indexed by tick rank and agent.

    ``T`` is the number of distinct ticks in ``df``; entries for missing
    (tick, agent) pairs are zero.
    """
    t = df["t"].to_numpy()
    a = df["agent"].to_numpy().astype(np.int64)
    _, t_idx = np.unique(t, return_inverse=True)
    n = n_agents_or_infer(df, n_agents)
    out = np.zeros((t_idx.max() + 1 if t_idx.size else 0, n), dtype=np.asarray(values).dtype)
    out[t_idx, a] = values
    return out


def entropy_bits(counts: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Plug-in Shannon entropy in bits of a count (or probability) array.

    With ``axis=None`` the whole array is one distribution; otherwise one
    distribution per slice along ``axis``. Empty distributions give 0.
    """
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, c / np.where(total > 0, total, 1.0), 0.0)
        terms = np.where(p > 0, -p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    h = terms.sum(axis=axis)
    return float(h) if axis is None else h


def entropy_nats_from_logp(logp: np.ndarray) -> np.ndarray:
    """Policy entropy in nats per row from a (M, K) array of log-probabilities.

    ``-inf`` entries are actions of probability zero and contribute nothing.
    """
    lp = np.asarray(logp, dtype=np.float64)
    finite = np.isfinite(lp)
    terms = np.where(finite, np.exp(np.where(finite, lp, 0.0)) * np.where(finite, lp, 0.0), 0.0)
    return -terms.sum(axis=1)


def null_summary(value: float, null: np.ndarray) -> dict[str, float]:
    """Pack an observed value and its permutation null into the standard dict.

    ``z`` is ``(value - mean) / sd`` and NaN when the null has zero spread.
    """
    null = np.asarray(null, dtype=np.float64)
    mean = float(null.mean()) if null.size else float("nan")
    sd = float(null.std(ddof=1)) if null.size > 1 else float("nan")
    z = (float(value) - mean) / sd if sd > 0 else float("nan")
    return {"value": float(value), "null_mean": mean, "null_sd": sd, "z": z}


def group_codes(df: pd.DataFrame, columns: tuple[str, ...] | list[str] | str) -> np.ndarray:
    """Dense integer code for the joint value of one or more columns."""
    cols = [columns] if isinstance(columns, str) else list(columns)
    if len(cols) == 1:
        return pd.factorize(df[cols[0]].to_numpy())[0].astype(np.int64)
    return df.groupby(cols, sort=False).ngroup().to_numpy().astype(np.int64)


def row_distributions(counts: np.ndarray) -> np.ndarray:
    """Normalise each row of a non-negative matrix to sum to one.

    Rows that sum to zero become uniform so that divergences stay defined.
    """
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum(axis=1, keepdims=True)
    uniform = np.full_like(c, 1.0 / c.shape[1])
    return np.where(tot > 0, c / np.where(tot > 0, tot, 1.0), uniform)


def jsd_bits(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence in bits between matched distributions; in [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return entropy_bits(m, axis=axis) - 0.5 * (entropy_bits(p, axis=axis) + entropy_bits(q, axis=axis))
