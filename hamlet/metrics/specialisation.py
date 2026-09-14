"""Specialisation metrics: division of labour, specialisation index, clustering.

Inputs are agent-by-task count matrices (ticks per zone per agent) built by
:func:`time_budget`, or per-agent feature rows for clustering. Entropies and
divergences are in bits.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from hamlet.config import N_ACTIONS, OBS, TRAIT_VECTOR_LEN, ZONE_NAMES
from hamlet.metrics.common import (
    DEFAULTS,
    after_burn_in,
    default_rng,
    entropy_bits,
    jsd_bits,
    n_agents_or_infer,
    null_summary,
    row_distributions,
)


# ---------------------------------------------------------------------------
# count matrices
# ---------------------------------------------------------------------------

def time_budget(
    df: pd.DataFrame,
    n_zones: int = len(ZONE_NAMES),
    burn_in_days: int | None = None,
    n_agents: int | None = None,
    active_only: bool = False,
    per_day: bool = False,
) -> np.ndarray:
    """Integer ticks each agent spent in each zone after burn-in.

    Returns an ``(N, n_zones)`` matrix, or ``(N, n_days, n_zones)`` with
    ``per_day`` (days after burn-in in order), which is the input the
    agent-day shuffle null of :func:`dol_indiv` needs. With ``active_only``
    only ticks where the activity was granted count.
    """
    data = after_burn_in(df, burn_in_days)
    n = n_agents_or_infer(data, n_agents)
    _, d_idx = np.unique(data["day"].to_numpy(), return_inverse=True)
    n_days = int(d_idx.max()) + 1 if d_idx.size else 0
    if active_only:
        keep = data["active"].to_numpy().astype(bool)
        data, d_idx = data[keep], d_idx[keep]
    a = data["agent"].to_numpy().astype(np.int64)
    z = data["zone"].to_numpy().astype(np.int64)
    counts = np.bincount((a * n_days + d_idx) * n_zones + z, minlength=n * n_days * n_zones)
    counts = counts.reshape(n, n_days, n_zones)
    return counts if per_day else counts.sum(axis=1)


# ---------------------------------------------------------------------------
# division of labour (Gorelick et al. 2004)
# ---------------------------------------------------------------------------

def _mutual_information_bits(counts: np.ndarray) -> tuple[float, float, float]:
    """(I(agent; task), H(agent), H(task)) in bits from a count matrix."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    h_agent = entropy_bits(c.sum(axis=1))
    h_task = entropy_bits(c.sum(axis=0))
    h_joint = entropy_bits(c)
    return h_agent + h_task - h_joint, h_agent, h_task


def dol_task_value(counts: np.ndarray) -> float:
    """DOL_task = I(agent; task) / H(task); 0 if every task is done in the same proportions by all."""
    mi, _, h_task = _mutual_information_bits(counts)
    return mi / h_task if h_task > 0 else float("nan")


def dol_indiv_value(counts: np.ndarray) -> float:
    """DOL_indiv = I(agent; task) / H(agent) in [0, 1] (Gorelick et al. 2004)."""
    mi, h_agent, _ = _mutual_information_bits(counts)
    return mi / h_agent if h_agent > 0 else float("nan")


def _pooled_multinomial_null(counts: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Resample each row from the pooled task distribution, keeping row totals.

    Returns an (n_perm, N, K) array of count matrices.
    """
    c = np.asarray(counts, dtype=np.int64)
    p = c.sum(axis=0).astype(np.float64)
    p = p / p.sum()
    totals = np.broadcast_to(c.sum(axis=1), (n_perm, c.shape[0]))
    return rng.multinomial(totals, p)


def _day_shuffle_null(day_counts: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Reassign whole agent-days to pseudo-agents at random, then sum over days.

    ``day_counts`` is (N, n_days, K). Every draw permutes the N * n_days
    agent-day rows and deals them out n_days at a time, so each pseudo-agent
    is a random collection of agent-days from the population. Returns an
    (n_perm, N, K) array of count matrices.
    """
    c = np.asarray(day_counts, dtype=np.int64)
    n, n_days, k = c.shape
    flat = c.reshape(n * n_days, k)
    out = np.empty((n_perm, n, k), dtype=np.int64)
    for i in range(n_perm):
        out[i] = flat[rng.permutation(n * n_days)].reshape(n, n_days, k).sum(axis=1)
    return out


DOL_NULLS = ("multinomial", "home_shuffle", "trait_and_home_shuffle")


def _null_matrices(
    counts: np.ndarray,
    null: str,
    n_perm: int,
    rng: np.random.Generator,
    replay_counts: Sequence[np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(observed (N, K) matrix, (n_perm, N, K) null matrices) for one null option."""
    c = np.asarray(counts, dtype=np.int64)
    if c.ndim not in (2, 3):
        raise ValueError("counts must be (N, K) or (N, n_days, K)")
    observed = c.sum(axis=1) if c.ndim == 3 else c
    if null == "multinomial":
        return observed, _pooled_multinomial_null(observed, n_perm, rng)
    if null not in DOL_NULLS:
        raise ValueError(f"unknown null {null!r}; choose from {DOL_NULLS}")
    if replay_counts is not None:
        return observed, np.stack([np.asarray(m, dtype=np.int64) for m in replay_counts])
    if null == "trait_and_home_shuffle":
        raise ValueError("the trait_and_home_shuffle null is built from replays: pass replay_counts from "
                         "hamlet.evaluate.trait_and_home_shuffle_counts")
    if c.ndim == 3:
        return observed, _day_shuffle_null(c, n_perm, rng)
    raise ValueError("the home_shuffle null needs replay_counts or an (N, n_days, K) count array; "
                     "use time_budget(..., per_day=True) or null='multinomial'")


def dol_indiv(
    counts: np.ndarray,
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    null: str = "home_shuffle",
    replay_counts: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """DOL_indiv against a null that removes the agent-task association.

    ``counts`` is the (N, K) matrix of :func:`time_budget`, or the (N, n_days, K)
    per-day version; the observed value always uses the sum over days.
    Two nulls are available:

    ``"multinomial"``
        Each agent's row is drawn from the pooled task proportions with its own
        row total. It keeps the agents' activity levels but treats every tick as
        an independent draw, so it is far too narrow whenever zone occupancy is
        autocorrelated (an agent stays in a zone for many ticks) and it knows
        nothing about geometry: a random walker's budget depends on where its
        home is, and that alone gives large z values.

    ``"home_shuffle"`` (default)
        The null is built from whole agent-days, so every agent-day keeps its
        internal composition and only its link to an agent label is broken. If
        ``replay_counts`` is given, those matrices are the null draws: they
        should come from replays of the same policy in which home tiles were
        permuted, with rows indexed so that the row label carries no identity
        (for example rows ordered by home slot), and then the observed excess
        is the part of the division of labour tied to the agent rather than to
        its home. Without ``replay_counts`` the per-day array is used: the
        N * n_days agent-days are permuted and dealt out n_days at a time to N
        pseudo-agents, recounted, and the index recomputed, ``n_perm`` times.
        This asks whether the division of labour is linked to the agent rather
        than to the day; a home effect much smaller than the day-to-day
        variation of an agent's budget is treated as noise by it.

    ``"trait_and_home_shuffle"``
        Replay-only: ``replay_counts`` must hold the count matrices of
        replays of the same policy on the same seed in which one permutation
        was applied to the trait vectors and the homes across agents (rows in
        agent-id order; see ``hamlet.evaluate.trait_and_home_shuffle_counts``).
        The null keeps every identity in place and moves the traits and homes
        between them, so the observed excess over it is the part of the
        division of labour tied to the identity rather than to the traits or
        the home it happened to carry. Without ``replay_counts`` it raises.

    Returns ``{"value", "null_mean", "null_sd", "z"}``.
    """
    rng = default_rng(rng)
    n_perm = DEFAULTS.n_perm if n_perm is None else int(n_perm)
    observed, draws = _null_matrices(counts, null, n_perm, rng, replay_counts)
    value = dol_indiv_value(observed)
    return null_summary(value, np.array([dol_indiv_value(m) for m in draws]))


def dol_task(
    counts: np.ndarray,
    n_perm: int | None = None,
    rng: np.random.Generator | None = None,
    null: str = "home_shuffle",
    replay_counts: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """DOL_task = I(agent; task) / H(task) with the same null options as :func:`dol_indiv`."""
    rng = default_rng(rng)
    n_perm = DEFAULTS.n_perm if n_perm is None else int(n_perm)
    observed, draws = _null_matrices(counts, null, n_perm, rng, replay_counts)
    value = dol_task_value(observed)
    return null_summary(value, np.array([dol_task_value(m) for m in draws]))


# ---------------------------------------------------------------------------
# Jensen-Shannon based indices (Mieczkowski et al. 2025)
# ---------------------------------------------------------------------------

def generalised_jsd(counts: np.ndarray) -> float:
    """JSD(P_1..P_N) = H(mean_i P_i) - mean_i H(P_i) in bits over row distributions."""
    p = row_distributions(counts)
    m = p.mean(axis=0)
    return float(entropy_bits(m) - entropy_bits(p, axis=1).mean())


def specialisation_index(counts: np.ndarray) -> float:
    """SI = generalised JSD / log2 N in [0, 1]; 0 identical generalists, 1 disjoint specialists."""
    n = np.asarray(counts).shape[0]
    if n < 2:
        return float("nan")
    return float(np.clip(generalised_jsd(counts) / np.log2(n), 0.0, 1.0))


def pairwise_jsd(counts: np.ndarray) -> np.ndarray:
    """(N, N) matrix of pairwise JSD in bits between row distributions; zero diagonal."""
    p = row_distributions(counts)
    return np.clip(jsd_bits(p[:, None, :], p[None, :, :], axis=-1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# clustering and cluster agreement
# ---------------------------------------------------------------------------

def silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette width from a precomputed distance matrix.

    Singleton clusters score 0. Returns NaN if fewer than two clusters.
    """
    d = np.asarray(distance, dtype=np.float64)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")
    n = d.shape[0]
    member = labels[:, None] == uniq[None, :]                 # (n, k)
    size = member.sum(axis=0)                                 # (k,)
    sums = d @ member.astype(np.float64)                      # (n, k) summed distance to each cluster
    own = member.argmax(axis=1)
    own_size = size[own]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(own_size > 1, sums[np.arange(n), own] / np.maximum(own_size - 1, 1), 0.0)
        mean_other = sums / size[None, :]
    mean_other[np.arange(n), own] = np.inf
    b = mean_other.min(axis=1)
    s = np.where(own_size > 1, (b - a) / np.maximum(np.maximum(a, b), np.finfo(float).tiny), 0.0)
    return float(s.mean())


def jsd_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance (sqrt of JSD in bits, a metric) between feature rows."""
    return np.sqrt(pairwise_jsd(features))


def cluster_agents(
    feature_matrix: np.ndarray | Sequence[np.ndarray],
    max_k: int | None = None,
) -> tuple[np.ndarray, float]:
    """Average-linkage clustering on JSD distance with silhouette-chosen k.

    ``feature_matrix`` has one non-negative row per agent (or agent-episode);
    a list of matrices is stacked. ``k`` ranges over ``2 .. max_k`` (default
    ``N - 1``). Returns 0-based labels and the silhouette of the chosen k.
    """
    x = np.vstack(feature_matrix) if isinstance(feature_matrix, (list, tuple)) else np.asarray(feature_matrix)
    n = x.shape[0]
    if n < 3:
        return np.zeros(n, dtype=np.int64), float("nan")
    d = jsd_distance_matrix(x)
    z = linkage(squareform(d, checks=False), method="average")
    k_max = min(max_k if max_k is not None else n - 1, n - 1)
    best_labels, best_score = np.zeros(n, dtype=np.int64), -np.inf
    for k in range(2, k_max + 1):
        labels = fcluster(z, k, criterion="maxclust") - 1
        score = silhouette(d, labels)
        if np.isfinite(score) and score > best_score:
            best_labels, best_score = labels.astype(np.int64), score
    return best_labels, float(best_score) if np.isfinite(best_score) else float("nan")


def _comb2(x: np.ndarray | float) -> np.ndarray | float:
    return x * (x - 1) / 2.0


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index between two labellings from the contingency table (Hubert and Arabie 1985)."""
    a = np.unique(np.asarray(labels_a), return_inverse=True)[1]
    b = np.unique(np.asarray(labels_b), return_inverse=True)[1]
    n = a.shape[0]
    table = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).reshape(a.max() + 1, b.max() + 1)
    index = _comb2(table.astype(np.float64)).sum()
    sum_a = _comb2(table.sum(axis=1).astype(np.float64)).sum()
    sum_b = _comb2(table.sum(axis=0).astype(np.float64)).sum()
    total = _comb2(float(n))
    expected = sum_a * sum_b / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return float((index - expected) / (max_index - expected))


def _pairwise_ari_matrix(labels: list[np.ndarray]) -> np.ndarray:
    m = len(labels)
    out = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            out[i, j] = out[j, i] = ari(labels[i], labels[j])
    return out


def ari_across_episodes(label_lists: Sequence[np.ndarray]) -> float:
    """Mean pairwise ARI over every pair of labellings of the same agents."""
    labels = list(label_lists)
    if len(labels) < 2:
        return float("nan")
    vals = [ari(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return float(np.mean(vals))


def ari_across_episodes_ci(
    label_lists: Sequence[np.ndarray],
    n_boot: int | None = None,
    rng: np.random.Generator | None = None,
    alpha: float | None = None,
) -> dict[str, float]:
    """Mean pairwise ARI with a within-seed percentile bootstrap interval over episodes.

    The episode, not the episode pair, is the resampling unit: every pair
    shares an episode with many other pairs, so resampling pairs treats
    dependent quantities as independent and gives an interval that is too
    narrow. Each draw resamples the episodes with replacement and takes the
    mean ARI over the pairs of distinct original episodes in the draw (a
    pair formed by two copies of the same episode has ARI 1 by construction
    and says nothing, so it is left out). Returns ``{"value", "ci_low",
    "ci_high", "n_episodes"}``; the interval is NaN when fewer than two
    distinct episodes can be drawn.
    """
    rng = default_rng(rng)
    n_boot = DEFAULTS.n_boot if n_boot is None else int(n_boot)
    alpha = DEFAULTS.alpha if alpha is None else float(alpha)
    labels = list(label_lists)
    m = len(labels)
    if m < 2:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_episodes": float(m)}
    matrix = _pairwise_ari_matrix(labels)
    iu = np.triu_indices(m, k=1)
    value = float(matrix[iu].mean())
    draws = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        sub = matrix[np.ix_(idx, idx)]
        distinct = idx[:, None] != idx[None, :]
        mask = np.triu(distinct, k=1)
        if mask.any():
            draws[b] = sub[mask].mean()
    ok = draws[np.isfinite(draws)]
    if ok.size == 0:
        return {"value": value, "ci_low": float("nan"), "ci_high": float("nan"), "n_episodes": float(m)}
    low, high = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return {"value": value, "ci_low": float(low), "ci_high": float(high), "n_episodes": float(m)}


# ---------------------------------------------------------------------------
# identity swap test
# ---------------------------------------------------------------------------

SWAP_KINDS = ("id", "traits", "both")


def swap_test(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    obs_batch: np.ndarray,
    n_agents: int,
    id_slice: slice | None = None,
    return_matrix: bool = False,
    swap: str = "id",
    trait_vectors: np.ndarray | None = None,
    trait_slice: slice | None = None,
) -> float | tuple[float, np.ndarray]:
    """Mean pairwise JSD between action distributions under swapped identities.

    ``policy_fn`` maps an (M, obs_dim) batch to (M, N_ACTIONS) action probabilities.
    For every agent k the identity of every observation is overwritten with
    agent k's; the JSD between the resulting distributions is averaged over
    the batch and over all unordered pairs of agents. ``swap`` says what an
    identity is: ``"id"`` overwrites the one-hot agent-ID block only (zero
    when the policy ignores that block), ``"traits"`` overwrites the trait
    block (``OBS["traits"]``) with row k of ``trait_vectors`` (``(N,
    TRAIT_VECTOR_LEN)``, the vectors the core wrote, so the test measures how
    much the policy conditions on traits), and ``"both"`` overwrites both.
    With traits in the observation the ID swap alone does not isolate
    identity: two agents differ by their traits as well as by their ID.
    """
    if swap not in SWAP_KINDS:
        raise ValueError(f"unknown swap {swap!r}; choose from {SWAP_KINDS}")
    sl = id_slice if id_slice is not None else OBS["agent_id"]
    tsl = trait_slice if trait_slice is not None else OBS["traits"]
    obs = np.asarray(obs_batch, dtype=np.float32)
    start = sl.start if sl.start is not None else 0
    if swap != "id":
        if trait_vectors is None:
            raise ValueError("swap='traits' and swap='both' need trait_vectors (N, TRAIT_VECTOR_LEN)")
        tv = np.asarray(trait_vectors, dtype=np.float32)
        if tv.shape != (n_agents, tsl.stop - tsl.start):
            raise ValueError(f"trait_vectors must be ({n_agents}, {tsl.stop - tsl.start}), got {tv.shape}")
    probs = np.empty((n_agents, obs.shape[0], N_ACTIONS), dtype=np.float64)
    for k in range(n_agents):
        o = obs.copy()
        if swap != "traits":
            o[:, start:start + n_agents] = 0.0
            o[:, start + k] = 1.0
        if swap != "id":
            o[:, tsl] = tv[k]
        probs[k] = np.asarray(policy_fn(o), dtype=np.float64)
    matrix = np.zeros((n_agents, n_agents), dtype=np.float64)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            matrix[i, j] = matrix[j, i] = float(jsd_bits(probs[i], probs[j], axis=-1).mean())
    iu = np.triu_indices(n_agents, k=1)
    mean = float(matrix[iu].mean()) if iu[0].size else 0.0
    return (mean, matrix) if return_matrix else mean
