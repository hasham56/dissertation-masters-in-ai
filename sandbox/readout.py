"""Post-hoc goal readout of one evaluation Parquet: ``goals.json``, one entry per agent, no LLM.

Command line::

    uv run python -m sandbox.readout PARQUET --out GOALS_JSON

``PARQUET`` is one evaluation episode in ``hamlet.config.LOG_COLUMNS``. This
tool runs unchanged on confirmatory-arm files under ``runs/`` (read-only: the
input is never modified and nothing is written beside it) and on sandbox
rollouts under ``sandbox/runs/``. ``GOALS_JSON`` must lie inside
``sandbox/runs/``.

What "goal" means here. Nothing is asked of the agent and no text is generated
by a model: the readout is a description of what the frozen policy did,
computed from the log by fixed rules and rendered by a fixed template. It is
the natural-language layer the study deliberately does without, kept post-hoc
and outside the analysis so that "the agent's routine" can be shown to a
reader without any goal text ever entering training or evaluation.

Per agent (days ``>= burn_in_days``; the zone column as logged):

- ``modal_zone_per_hour_bin``: the most frequent zone in each of the eight
  3-hour bins (``hour_bin = t_day // 30``, from ``routine.add_bins``) with its
  share of the bin's ticks;
- ``windows``: ``wake`` (median over days of the first tick of the day at
  which the agent is not at HOME) and ``sleep`` (median over days of the
  first tick of the HOME run that lasts to the end of the day; absent when
  the agent is not home at day end on most days); ``work`` (the longest run
  of active FARM or OFFICE ticks per day; median start and end over the days
  that have one at least ``MIN_WINDOW_TICKS`` long; absent when fewer than
  half the days do); ``meals`` (starts of active CANTEEN runs and of active
  MARKET ticks, clustered into at most ``MAX_MEALS_PER_DAY`` daily meal
  times, each with its median tick and the fraction of days it occurs);
  ``social`` (the longest run of active SOCIAL ticks per day, as for work).
  Ticks are ticks of day (0..239) and are also given as ``HH:MM``;
- ``schedule``: one sentence assembled by ``schedule_sentence`` from the
  windows; a window that is absent is left out, never guessed.

Output ``goals.json``: ``{"source": {...}, "agents": {"0": {...}, ...},
"exploratory": true}``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from hamlet.config import CANTEEN, FARM, HOME, MARKET, OFFICE, SOCIAL, ZONE_NAMES
from hamlet.metrics import routine
from hamlet.metrics.common import after_burn_in, ticks_per_day_or_default

from sandbox import EXPLORATORY_TAG, assert_sandbox_output

MIN_WINDOW_TICKS = 5      # runs shorter than this (30 game minutes) are not windows
MAX_MEALS_PER_DAY = 3
MEAL_CLUSTER_TICKS = 30   # meal starts within three hours of each other are one meal time
TICKS_PER_HOUR = 10
N_HOUR_BINS = 8
MIN_DAY_FRACTION = 0.5    # a window must occur on at least this fraction of days to be reported


def tick_to_clock(t_day: int) -> str:
    """``t_day`` (0..239) to ``"HH:MM"`` at 6 game minutes per tick."""
    t = int(round(t_day))
    hours, ticks = divmod(t, TICKS_PER_HOUR)
    return f"{hours % 24:02d}:{ticks * 6:02d}"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, end)]`` of the maximal runs of True in ``mask``; ``end`` is exclusive."""
    if mask.size == 0:
        return []
    padded = np.r_[False, mask.astype(bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _longest_run(mask: np.ndarray, minimum: int) -> Optional[tuple[int, int]]:
    runs = [r for r in _runs(mask) if r[1] - r[0] >= minimum]
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def _day_arrays(df_agent: pd.DataFrame, T: int) -> tuple[np.ndarray, np.ndarray]:
    """``zone (days, T)`` and ``active (days, T)`` of one agent, ordered by day and tick of day."""
    sub = df_agent.sort_values(["day", "t_day"])
    days = np.unique(sub["day"].to_numpy())
    zone = np.full((len(days), T), -1, dtype=np.int64)
    active = np.zeros((len(days), T), dtype=bool)
    d_idx = np.searchsorted(days, sub["day"].to_numpy())
    zone[d_idx, sub["t_day"].to_numpy()] = sub["zone"].to_numpy()
    active[d_idx, sub["t_day"].to_numpy()] = sub["active"].to_numpy().astype(bool)
    return zone, active


def _window_summary(runs: list[Optional[tuple[int, int]]], n_days: int, zones: Optional[list[int]] = None) -> Optional[dict[str, Any]]:
    present = [r for r in runs if r is not None]
    if not present or len(present) < MIN_DAY_FRACTION * n_days:
        return None
    starts = np.array([r[0] for r in present])
    ends = np.array([r[1] for r in present])
    out: dict[str, Any] = {
        "start_tick": float(np.median(starts)), "end_tick": float(np.median(ends)),
        "start": tick_to_clock(np.median(starts)), "end": tick_to_clock(np.median(ends)),
        "start_spread_ticks": float(np.percentile(starts, 75) - np.percentile(starts, 25)),
        "days_present": len(present), "days": int(n_days),
    }
    if zones:
        values, counts = np.unique(np.asarray(zones), return_counts=True)
        out["zone"] = ZONE_NAMES[int(values[np.argmax(counts)])]
    return out


def modal_zone_per_hour_bin(df_agent: pd.DataFrame) -> list[dict[str, Any]]:
    """Eight entries: ``{"hour_bin", "hours", "zone", "share"}`` for one agent's rows."""
    binned = routine.add_bins(df_agent)
    out = []
    for k in range(N_HOUR_BINS):
        zones = binned.loc[binned["hour_bin"] == k, "zone"].to_numpy().astype(np.int64)
        if zones.size == 0:
            out.append({"hour_bin": k, "hours": f"{3 * k:02d}:00-{3 * k + 3:02d}:00", "zone": None, "share": 0.0})
            continue
        counts = np.bincount(zones, minlength=len(ZONE_NAMES))
        z = int(np.argmax(counts))
        out.append({"hour_bin": k, "hours": f"{3 * k:02d}:00-{3 * k + 3:02d}:00", "zone": ZONE_NAMES[z],
                    "share": float(counts[z] / zones.size)})
    return out


def infer_windows(df_agent: pd.DataFrame, ticks_per_day: Optional[int] = None) -> dict[str, Any]:
    """The ``wake``, ``sleep``, ``work``, ``meals`` and ``social`` windows of one agent."""
    T = ticks_per_day_or_default(ticks_per_day)
    zone, active = _day_arrays(df_agent, T)
    n_days = zone.shape[0]
    wake_ticks, sleep_ticks = [], []
    work_runs, work_zones, social_runs = [], [], []
    meal_starts: list[int] = []
    for d in range(n_days):
        z, a = zone[d], active[d]
        away = np.flatnonzero(z != HOME)
        wake_ticks.append(int(away[0]) if away.size else None)
        home_runs = _runs(z == HOME)
        last = home_runs[-1] if home_runs else None
        sleep_ticks.append(int(last[0]) if last is not None and last[1] == T and last[0] > 0 else None)
        job = a & ((z == FARM) | (z == OFFICE))
        run = _longest_run(job, MIN_WINDOW_TICKS)
        work_runs.append(run)
        if run is not None:
            work_zones.extend(z[run[0]:run[1]].tolist())
        social_runs.append(_longest_run(a & (z == SOCIAL), MIN_WINDOW_TICKS))
        meal_starts.extend(r[0] for r in _runs(a & (z == CANTEEN)))
        meal_starts.extend(r[0] for r in _runs(a & (z == MARKET)))

    def median_or_none(values: list[Optional[int]]) -> Optional[dict[str, Any]]:
        present = [v for v in values if v is not None]
        if not present or len(present) < MIN_DAY_FRACTION * n_days:
            return None
        med = float(np.median(present))
        return {"tick": med, "clock": tick_to_clock(med),
                "spread_ticks": float(np.percentile(present, 75) - np.percentile(present, 25)),
                "days_present": len(present), "days": int(n_days)}

    meals: list[dict[str, Any]] = []
    if meal_starts:
        starts = np.sort(np.asarray(meal_starts))
        clusters: list[list[int]] = [[int(starts[0])]]
        for s in starts[1:]:
            if s - clusters[-1][-1] <= MEAL_CLUSTER_TICKS:
                clusters[-1].append(int(s))
            else:
                clusters.append([int(s)])
        clusters.sort(key=len, reverse=True)
        for c in clusters[:MAX_MEALS_PER_DAY]:
            if len(c) < MIN_DAY_FRACTION * n_days:
                continue
            med = float(np.median(c))
            meals.append({"tick": med, "clock": tick_to_clock(med), "days_present": min(len(c), n_days),
                          "days": int(n_days), "starts_per_day": len(c) / max(n_days, 1)})
        meals.sort(key=lambda m: m["tick"])

    return {
        "wake": median_or_none(wake_ticks),
        "sleep": median_or_none(sleep_ticks),
        "work": _window_summary(work_runs, n_days, work_zones),
        "meals": meals,
        "social": _window_summary(social_runs, n_days),
        "days": int(n_days),
    }


def schedule_sentence(agent: int, windows: dict[str, Any]) -> str:
    """Assemble the one-line schedule from the windows by template; no model, no free text."""
    parts: list[str] = []
    if windows.get("wake"):
        parts.append(f"is up at {windows['wake']['clock']}")
    if windows.get("work"):
        w = windows["work"]
        place = {"FARM": "the farm", "OFFICE": "the office"}.get(w.get("zone", ""), "a job")
        parts.append(f"works at {place} {w['start']} to {w['end']}")
    meals = windows.get("meals") or []
    if meals:
        clocks = [m["clock"] for m in meals]
        parts.append("eats around " + (clocks[0] if len(clocks) == 1 else ", ".join(clocks[:-1]) + " and " + clocks[-1]))
    if windows.get("social"):
        s = windows["social"]
        parts.append(f"socialises {s['start']} to {s['end']}")
    if windows.get("sleep"):
        parts.append(f"is home for the night by {windows['sleep']['clock']}")
    if not parts:
        return f"Agent {agent} shows no regular window over {windows.get('days', 0)} days."
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Agent {agent} {body}."


def read_goals(df: pd.DataFrame, burn_in_days: Optional[int] = None, path: Optional[Path] = None) -> dict[str, Any]:
    """The ``goals.json`` dictionary for one evaluation episode; JSON-serialisable."""
    data = after_burn_in(df, burn_in_days)
    first = df.iloc[0]
    agents: dict[str, Any] = {}
    for agent in sorted(int(a) for a in data["agent"].unique()):
        sub = data[data["agent"] == agent]
        windows = infer_windows(sub)
        agents[str(agent)] = {
            "modal_zone_per_hour_bin": modal_zone_per_hour_bin(sub),
            "windows": windows,
            "schedule": schedule_sentence(agent, windows),
        }
    return {
        "source": {"path": str(path) if path is not None else None, "condition": str(first["condition"]),
                   "seed": int(first["seed"]), "checkpoint": str(first["checkpoint"]),
                   "days_used": sorted(int(d) for d in data["day"].unique())},
        "agents": agents,
        **EXPLORATORY_TAG,
    }


def write_goals(parquet: Path, out: Path) -> Path:
    """``read_goals`` on ``parquet`` written to ``out`` (inside ``sandbox/runs/``); the input is opened read-only."""
    out = assert_sandbox_output(Path(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet, engine="pyarrow")
    out.write_text(json.dumps(read_goals(df, path=Path(parquet)), indent=2))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path, help="one evaluation episode (LOG_COLUMNS); may be a confirmatory file under runs/, read-only")
    parser.add_argument("--out", type=Path, required=True, help="goals.json path, inside sandbox/runs/")
    args = parser.parse_args(argv)
    out = write_goals(args.parquet, args.out)
    goals = json.loads(out.read_text())
    for agent, entry in goals["agents"].items():
        print(entry["schedule"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
