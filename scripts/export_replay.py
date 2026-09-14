"""Export one evaluation Parquet as a compact replay JSON for the viewer.

    uv run python scripts/export_replay.py runs/A-S1-N9/seed0/eval/stochastic/A-S1-N9/ckpt_0864/seed10000_ep0.parquet
    uv run python scripts/export_replay.py a.parquet b.parquet --outdir viz/replays

Each Parquet becomes ``<outdir>/<condition>[_seed<k>]_<run label>_eval<eval seed>.json``, and
``<outdir>/manifest.json``, which the viewer reads, is rebuilt from every replay in that directory, so
replays exported one command at a time accumulate.

One record per tick per agent: ``t, day, t_day, agent, zone, active, queued, E, F, C, dest,
journey_start``. ``zone`` is the zone name from the log. While the agent is in TRANSIT, ``dest`` is
the zone named by the held action (the ``action`` column carries the held action during a journey,
so it is read straight off the row) and ``journey_start`` is the tick the journey began; both are
null otherwise. Journeys are ``HamletConfig.travel_ticks`` ticks long and the constant is read from
the config, never typed in.

Read-only: nothing under ``runs/`` is written or modified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamlet.config import ACTION_NAMES, ZONE_NAMES  # noqa: E402

TRANSIT = ZONE_NAMES.index("TRANSIT")
# The zone each action requests; IDLE requests none.
ACTION_ZONE = {i: (n.removeprefix("GO_") if n != "IDLE" else None) for i, n in enumerate(ACTION_NAMES)}

# ---- the world the run was trained in, from its own manifest -----------------------------------
# A replay is read by a viewer that has no other way to know how big the village was or what its
# zones were called. Those facts belong to the run, not to whatever ``hamlet/config.py`` holds when
# the export happens: the canteen's printed name changed with the N = 9 rerun, and reading it
# from today's config would relabel the archived N = 8 replays. So every world value in
# a replay is read back from the run's own metadata.json, and an export with no manifest is refused.
CANTEEN_NAME_BY_POPULATION = {8: "Restaurant"}       # the N = 8 study's figures, tables and replays
CANTEEN_NAME_DEFAULT = "Food Street"                 # the N = 9 world and later


def find_manifest(parquet: Path) -> Path:
    """The metadata.json of the run this Parquet belongs to.

    A trained cell holds its own (``<condition>/seed<k>/metadata.json``). A scripted baseline does
    not: it lives at ``<condition>/GREEDY-CLOCK/`` and was rolled out in the same world as the
    cells beside it, so the condition's own cells answer for it. Anything else is refused.
    """
    for d in [parquet.parent, *parquet.parents]:
        own = d / "metadata.json"
        if own.is_file():
            return own
        siblings = sorted(d.glob("seed*/metadata.json"))
        if siblings:
            return siblings[0]
    raise SystemExit(
        f"no metadata.json for {parquet}: a replay carries the world it was run in, and that is "
        f"read from the run's manifest. Export refused.")


def world_from_manifest(manifest: Path) -> dict[str, Any]:
    """The population, zone capacities and printed names the run was trained with."""
    cfg = (json.loads(manifest.read_text()).get("hamlet_config")
           or json.loads(manifest.read_text()).get("config") or {})
    need = ("n_agents", "ticks_per_day", "n_days", "burn_in_days", "travel_ticks", "night")
    missing = [k for k in need if cfg.get(k) is None]
    if missing:
        raise SystemExit(f"{manifest} does not record {missing}; export refused")
    n = int(cfg["n_agents"])
    # The capacity rules are the simulation's own and scale with N; the manifest stores None for
    # each, meaning "derived", so they are derived here from the population it does record.
    cap_job, cap_canteen = -(-n // 2), -(-n // 4)
    capacities = {
        "HOME": 1, "FARM": cap_job, "OFFICE": cap_job, "CANTEEN": cap_canteen,
        "SOCIAL": 0, "MARKET": 0, "TRANSIT": 0,          # 0 means unlimited in the viewer
    }
    # World v2-lite staffed the bar. Its company gain is
    #     social_gain * (bar_solo_share + bar_company_step * min(others, bar_company_cap))
    # so bar_solo_share is the floor a lone drinker earns and bar_company_step the slope company
    # adds. A staffed bar is never empty, which is why the viewer draws somebody behind the counter.
    # World v1 records neither flag and has no staff: the manifest's own flags decide, not a
    # version number and not today's config.
    bar = {
        "staffed": "bar_solo_share" in cfg,
        "floor": cfg.get("bar_solo_share"),
        "slope": cfg.get("bar_company_step"),
    }
    canteen = CANTEEN_NAME_BY_POPULATION.get(n, CANTEEN_NAME_DEFAULT)
    display = {"HOME": "Home", "FARM": "Farm", "OFFICE": "Office", "CANTEEN": canteen,
               "SOCIAL": "Social", "MARKET": "Market", "TRANSIT": "Transit"}
    return {
        "n_agents": n,
        "ticks_per_day": int(cfg["ticks_per_day"]),
        "n_days": int(cfg["n_days"]),
        "burn_in_days": int(cfg["burn_in_days"]),
        "travel_ticks": int(cfg["travel_ticks"]),
        "night": [int(cfg["night"][0]), int(cfg["night"][1])],
        "capacities": capacities,
        "display_names": display,
        "bar": bar,
        "manifest": str(manifest),
    }


def journeys(df: pd.DataFrame, travel_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """``(dest, journey_start)`` per row: the held destination and the tick the journey began.

    A journey is a maximal run of TRANSIT ticks within one agent. Its destination is the zone the
    held action names, which the log already carries on every tick of the journey.
    """
    n = len(df)
    dest = np.full(n, None, dtype=object)
    start = np.full(n, -1, dtype=np.int64)
    for _, idx in df.groupby("agent", sort=False).groups.items():
        g = df.loc[idx].sort_values("t")
        pos = g.index.to_numpy()
        in_transit = (g["zone"].to_numpy() == TRANSIT)
        t = g["t"].to_numpy()
        act = g["action"].to_numpy()
        run_start = None
        for k in range(len(g)):
            if in_transit[k] and run_start is None:
                run_start = t[k]
            if in_transit[k]:
                dest[df.index.get_indexer([pos[k]])[0]] = ACTION_ZONE.get(int(act[k]))
                start[df.index.get_indexer([pos[k]])[0]] = run_start
            else:
                run_start = None
    return dest, start


def export(parquet: Path, out: Path, label: str = "") -> dict[str, Any]:
    world = world_from_manifest(find_manifest(parquet))
    df = pd.read_parquet(parquet).reset_index(drop=True)
    df = df.sort_values(["t", "agent"]).reset_index(drop=True)
    dest, start = journeys(df, world["travel_ticks"])

    zone_names = [ZONE_NAMES[z] for z in df["zone"].to_numpy()]
    records = {
        "label": label,
        "source": str(parquet),
        "condition": str(df["condition"].iloc[0]),
        "checkpoint": str(df["checkpoint"].iloc[0]),
        "eval_seed": int(df["seed"].iloc[0]),
        "manifest": world["manifest"],
        "n_agents": world["n_agents"],
        "ticks_per_day": world["ticks_per_day"],
        "n_days": world["n_days"],
        "burn_in_days": world["burn_in_days"],
        "travel_ticks": world["travel_ticks"],
        "night": world["night"],
        "capacities": world["capacities"],
        "display_names": world["display_names"],
        "bar": world["bar"],
        "zone_names": list(ZONE_NAMES),
        "columns": ["t", "day", "t_day", "agent", "zone", "active", "queued", "E", "F", "C",
                    "dest", "journey_start"],
        "rows": [
            [int(r.t), int(r.day), int(r.t_day), int(r.agent), zone_names[i],
             bool(r.active), bool(r.queued), round(float(r.E), 4), round(float(r.F), 4),
             round(float(r.C), 4), dest[i], (int(start[i]) if start[i] >= 0 else None)]
            for i, r in enumerate(df.itertuples(index=False))
        ],
    }
    logged = int(df["agent"].max()) + 1
    if logged != world["n_agents"]:
        raise SystemExit(f"{parquet} logs {logged} agents but {world['manifest']} says "
                         f"{world['n_agents']}; export refused")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, separators=(",", ":")))
    return records


def check(records: dict[str, Any]) -> list[str]:
    """Sanity checks on the exported replay; returns a list of complaints.

    A journey is exactly ``travel_ticks`` ticks unless the episode truncates mid-journey, which
    cuts the last run short for any agent still travelling on the final tick. Those are reported
    separately from genuinely wrong lengths, because only the latter is a defect.
    """
    tt = records["travel_ticks"]
    cols = records["columns"]
    ti, ai, zi = cols.index("t"), cols.index("agent"), cols.index("zone")
    per_agent: dict[int, list] = {}
    last_t = 0
    for r in records["rows"]:
        per_agent.setdefault(r[ai], []).append(r)
        last_t = max(last_t, r[ti])
    wrong, censored = [], 0
    for agent, rows in per_agent.items():
        rows.sort(key=lambda r: r[ti])
        run, start_t = 0, None
        for r in rows:
            if r[zi] == "TRANSIT":
                if run == 0:
                    start_t = r[ti]
                run += 1
            else:
                if run and run != tt:
                    wrong.append((agent, start_t, run))
                run = 0
        if run:                                   # still travelling on the final tick
            if start_t + run - 1 == last_t:
                censored += 1
            else:
                wrong.append((agent, start_t, run))
    out = []
    if wrong:
        out.append(f"{len(wrong)} journey(s) not {tt} ticks and not truncated by the episode end: {wrong}")
    if censored:
        out.append(f"{censored} journey(s) cut short by the episode ending at tick {last_t} (expected)")
    return out


def replay_id(parquet: Path, rec: dict[str, Any]) -> str:
    """``<condition>[_seed<k>]_<run label>_eval<eval seed>``: unique across conditions, seeds and policies."""
    train = next((part for part in reversed(parquet.parts[:-1]) if re.fullmatch(r"seed\d+", part)), None)
    return "_".join(x for x in (rec["condition"], train, parquet.parent.name, f"eval{rec['eval_seed']}") if x)


def write_manifest(outdir: Path) -> list[dict[str, Any]]:
    """``manifest.json`` for the viewer, listing every replay in ``outdir``."""
    entries = []
    for f in sorted(outdir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        r = json.loads(f.read_text())
        # Mean energy, satiety and social over the days after warm-up: the viewer turns them into a
        # one-word description of the replay (thriving, lonely, exhausted, hungry) for its picker.
        ix = {c: i for i, c in enumerate(r["columns"])}
        kept = [row for row in r["rows"] if row[ix["day"]] >= r["burn_in_days"]]
        needs = [round(float(np.mean([row[ix[k]] for row in kept])), 3) for k in ("E", "F", "C")] if kept else None
        entries.append({"id": f.stem, "file": f.name, "label": r["label"], "condition": r["condition"],
                        "checkpoint": r["checkpoint"], "n_agents": r["n_agents"],
                        "canteen_name": r["display_names"]["CANTEEN"], "eval_seed": r["eval_seed"],
                        "ticks": len(r["rows"]) // r["n_agents"], "needs": needs})
    (outdir / "manifest.json").write_text(json.dumps({"replays": entries}, indent=2))
    return entries


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+", help="evaluation Parquets, one replay each")
    ap.add_argument("--outdir", default="viz/replays")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    for path in map(Path, args.parquets):
        tmp = outdir / ".export.json"
        rec = export(path, tmp)
        name = replay_id(path, rec)
        rec["label"] = f"{rec['condition']}, {path.parent.name}, evaluation seed {rec['eval_seed']}"
        out = outdir / f"{name}.json"
        out.write_text(json.dumps(rec, separators=(",", ":")))
        tmp.unlink()
        print(f"  wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(rec['rows']):,} rows)")
        for problem in check(rec):
            print(f"      note: {problem}")
    entries = write_manifest(outdir)
    print(f"  wrote {outdir / 'manifest.json'} ({len(entries)} replay(s))")


if __name__ == "__main__":
    main()
