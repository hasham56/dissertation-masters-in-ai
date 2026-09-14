"""Degeneracy flags and a one-row summary per evaluation episode.

Flagged runs are reported, never dropped. The summary is what the results
tables are built from; everything else is in the other metric modules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hamlet.config import LOG_COLUMNS, ZONE_NAMES
from hamlet.metrics.common import DEFAULTS, after_burn_in, entropy_nats_from_logp, n_agents_or_infer
from hamlet.metrics.social import contention_per_agent_day, gossip_curve
from hamlet.metrics.specialisation import time_budget
from hamlet.traits import CSV_COLUMNS, read_traits_csv

_LOGP_COLUMNS = [c for c in LOG_COLUMNS if c.startswith("logp")]
_STATE_COLUMNS = ("E", "F", "C", "W", "M", "D")
_ID_COLUMNS = ("seed", "condition", "checkpoint", "episode")
_TRAIT_COLUMNS = tuple(c for c in CSV_COLUMNS if c not in ("agent_id", "variant"))
_POPULATION_TAG = "-pop_"


def population_of(condition: str) -> str:
    """Population name encoded in a condition name (``-pop_<name>`` suffix), ``neutral`` if absent."""
    condition = str(condition)
    if _POPULATION_TAG in condition:
        return condition.split(_POPULATION_TAG, 1)[1]
    return "neutral"


def mean_policy_entropy(df: pd.DataFrame, burn_in_days: int | None = None) -> float:
    """Mean policy entropy in nats from the log-probability columns; NaN if absent."""
    data = after_burn_in(df, burn_in_days)
    if not set(_LOGP_COLUMNS).issubset(data.columns):
        return float("nan")
    lp = data[_LOGP_COLUMNS].to_numpy(dtype=np.float64)
    ok = ~np.isnan(lp).any(axis=1)
    if not ok.any():
        return float("nan")
    return float(entropy_nats_from_logp(lp[ok]).mean())


def mean_return(df: pd.DataFrame) -> float:
    """Mean over agents of the episode return (sum of reward over every tick)."""
    return float(df.groupby("agent")["reward"].sum().mean())


def median_distinct_zones_per_day(df: pd.DataFrame, burn_in_days: int | None = None) -> float:
    """Median over agent-days of the number of distinct zones visited."""
    data = after_burn_in(df, burn_in_days)
    if data.empty:
        return float("nan")
    return float(data.groupby(["agent", "day"])["zone"].nunique().median())


def max_zone_share(df: pd.DataFrame, burn_in_days: int | None = None) -> float:
    """Largest fraction of post-burn-in ticks any agent spends in a single zone."""
    counts = time_budget(df, burn_in_days=burn_in_days)
    tot = counts.sum(axis=1)
    if not (tot > 0).any():
        return float("nan")
    return float((counts.max(axis=1)[tot > 0] / tot[tot > 0]).max())


def flags(
    df: pd.DataFrame,
    greedy_return: float | None = None,
    min_zones_per_day: int | None = None,
    max_share: float | None = None,
    min_entropy_nats: float | None = None,
    burn_in_days: int | None = None,
) -> dict[str, float | bool]:
    """Degeneracy flags for one episode plus the numbers behind them.

    ``few_zones``: median distinct zones per agent-day below ``min_zones_per_day``.
    ``single_zone``: some agent spends more than ``max_share`` of its ticks in
    one zone. ``low_entropy``: mean policy entropy below ``min_entropy_nats``
    (only when log-probabilities are logged). ``below_greedy``: mean episode
    return below ``greedy_return`` (same unit as :func:`mean_return`) when
    given. ``degenerate`` is the disjunction.
    """
    min_zones = DEFAULTS.min_zones_per_day if min_zones_per_day is None else int(min_zones_per_day)
    share = DEFAULTS.max_zone_share if max_share is None else float(max_share)
    ent = DEFAULTS.min_entropy_nats if min_entropy_nats is None else float(min_entropy_nats)

    zones = median_distinct_zones_per_day(df, burn_in_days)
    top = max_zone_share(df, burn_in_days)
    entropy = mean_policy_entropy(df, burn_in_days)
    ret = mean_return(df)

    few_zones = bool(np.isfinite(zones) and zones < min_zones)
    single_zone = bool(np.isfinite(top) and top > share)
    low_entropy = bool(np.isfinite(entropy) and entropy < ent)
    below_greedy = bool(greedy_return is not None and np.isfinite(ret) and ret < greedy_return)
    return {
        "degenerate": few_zones or single_zone or low_entropy or below_greedy,
        "few_zones": few_zones,
        "single_zone": single_zone,
        "low_entropy": low_entropy,
        "below_greedy": below_greedy,
        "median_distinct_zones": zones,
        "max_zone_share": top,
        "mean_entropy_nats": entropy,
        "mean_return": ret,
    }


def _summary_row(df: pd.DataFrame, burn_in_days: int | None, traits_file: Path | None = None) -> dict[str, float | str]:
    data = after_burn_in(df, burn_in_days)
    row: dict[str, float | str] = {}
    for col in _ID_COLUMNS:
        if col in df.columns:
            row[col] = df[col].iloc[0]
    row["population"] = population_of(df["condition"].iloc[0]) if "condition" in df.columns else "neutral"
    row["n_agents"] = n_agents_or_infer(df, None)
    if traits_file is not None and traits_file.exists():
        traits, variant = read_traits_csv(traits_file)
        row["population"] = variant
        rows = [t.as_row(i, variant) for i, t in enumerate(traits)]
        for col in _TRAIT_COLUMNS:
            row[f"mean_{col}"] = float(np.mean([r[col] for r in rows])) if rows else float("nan")
    if "effort_penalty" in df.columns:
        row["mean_effort_penalty"] = float(data["effort_penalty"].mean()) if not data.empty else float("nan")
    counts = time_budget(df, burn_in_days=burn_in_days)
    frac = counts.sum(axis=0) / max(counts.sum(), 1)
    for z, name in enumerate(ZONE_NAMES):
        row[f"frac_{name}"] = float(frac[z]) if z < frac.size else 0.0
    for col in _STATE_COLUMNS:
        row[f"mean_{col}"] = float(data[col].mean()) if not data.empty else float("nan")
    row["contention_per_agent_day"] = contention_per_agent_day(df, burn_in_days)
    curve = gossip_curve(df)
    row["gossip_fraction_end"] = float(curve[-1]) if curve.size else float("nan")
    row["mean_return"] = mean_return(df)
    row["mean_entropy_nats"] = mean_policy_entropy(df, burn_in_days)
    return row


def traits_file_for(parquet: Path) -> Path | None:
    """The traits CSV beside an episode log: ``traits_seed<seed>.csv`` (evaluation) or ``traits.csv`` (training)."""
    stem = parquet.stem
    if stem.startswith("seed") and "_ep" in stem:
        candidate = parquet.with_name(f"traits_{stem.split('_ep', 1)[0]}.csv")
        if candidate.exists():
            return candidate
    candidate = parquet.with_name("traits.csv")
    return candidate if candidate.exists() else None


def summary(
    source: pd.DataFrame | str | Path,
    burn_in_days: int | None = None,
    traits_file: Path | None = None,
) -> pd.DataFrame:
    """One row per episode: time-budget fractions, mean states and drive, contention, gossip.

    ``source`` is an episode DataFrame (one row out) or a run directory whose
    Parquet files are each summarised (one row per file, sorted by name).
    Every row carries the population name; when a traits CSV lies beside a
    log (``traits_seed<seed>.csv`` or ``traits.csv``, found by
    :func:`traits_file_for`) the population-mean of every trait is joined as
    ``mean_<trait>``. For a DataFrame source pass the CSV as ``traits_file``
    to get the same join.
    """
    if isinstance(source, pd.DataFrame):
        return pd.DataFrame([_summary_row(source, burn_in_days, traits_file)])
    files = sorted(Path(source).glob("*.parquet"))
    rows = [_summary_row(pd.read_parquet(f), burn_in_days, traits_file_for(f)) for f in files]
    return pd.DataFrame(rows)
