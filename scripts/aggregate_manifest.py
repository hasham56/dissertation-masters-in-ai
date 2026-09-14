"""Reproducibility manifest: one row per training run under a run root, plus energy totals.

    uv run python scripts/aggregate_manifest.py                 # runs/ -> runs/manifest.csv
    uv run python scripts/aggregate_manifest.py --root runs/_smoke_post --out runs/_smoke_post/manifest.csv

Columns, one row per ``<root>/<condition>/seed<s>/`` that holds ``metadata.json``:
condition, seed, trainer, threads, n_envs, step_budget, agent_steps, updates, git_describe,
git_commit, started_utc, finished_utc, wall_clock_min, agent_steps_per_s, energy_kwh,
cpu_energy_kwh, emissions_kg (from the run's own ``emissions.json`` / ``metadata.energy``),
selected_checkpoint, selected_iqm_return, status, plus the degeneracy flags when
``<run_dir>/flags.json`` exists (written by ``scripts/report_grid.py --section flags``):
flag_few_zones, flag_single_zone, flag_low_entropy, flag_below_greedy, flag_below_greedy_iqm, degenerate.
Energy totals (kWh, kg CO2e, wall-clock hours) are printed for the rows that carry them.
Reads only; every number comes from the files the trainer and the launcher wrote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FLAG_KEYS = ("few_zones", "single_zone", "low_entropy", "below_greedy", "below_greedy_iqm", "degenerate")


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def run_row(run_dir: Path) -> Optional[dict[str, Any]]:
    """The manifest row of one run directory, or None when it holds no ``metadata.json``."""
    meta = _load(run_dir / "metadata.json")
    if not meta:
        return None
    manifest = _load(run_dir / "manifest.json")
    selection = _load(run_dir / "selection.json")
    energy = meta.get("energy") or _load(run_dir / "emissions.json") or {}
    selected = selection.get("selected")
    sel_iqm = next((c.get("iqm_return") for c in selection.get("candidates", []) if c.get("path") == selected), None)
    flags = _load(run_dir / "flags.json")
    git = meta.get("git") or manifest.get("git") or {}
    hyper = meta.get("hyperparameters") or manifest.get("hyperparameters") or {}
    row = {
        "condition": meta.get("condition"), "seed": meta.get("env_seed"),
        "trainer": meta.get("trainer"), "threads": meta.get("threads", manifest.get("threads")),
        "n_envs": meta.get("n_envs"), "step_budget": meta.get("step_budget", hyper.get("total_agent_steps")),
        "agent_steps": meta.get("agent_steps"), "updates": meta.get("updates"),
        "git_describe": git.get("describe"), "git_commit": git.get("commit"),
        "started_utc": meta.get("started_utc"), "finished_utc": meta.get("finished_utc"),
        "wall_clock_min": (meta["wall_clock_s"] / 60) if meta.get("wall_clock_s") is not None else None,
        "agent_steps_per_s": meta.get("agent_steps_per_s"),
        "energy_kwh": energy.get("energy_kwh"), "cpu_energy_kwh": energy.get("cpu_energy_kwh"),
        "emissions_kg": energy.get("emissions_kg", meta.get("emissions_kg")),
        "selected_checkpoint": Path(selected).name if selected else None, "selected_iqm_return": sel_iqm,
        "status": meta.get("status"), "run_dir": str(run_dir),
    }
    for key in FLAG_KEYS:
        row[f"flag_{key}" if key != "degenerate" else "degenerate"] = flags.get(key) if flags else None
    return row


def aggregate(root: Path) -> pd.DataFrame:
    rows = [run_row(p.parent) for p in sorted(root.glob("*/seed*/metadata.json"))]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows).sort_values(["condition", "seed"]).reset_index(drop=True) if rows else pd.DataFrame()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="run root to read and write under. No default, deliberately: a default of\n                             'runs' let a half-analysed rerun be read into a finished report.")
    parser.add_argument("--out", default=None, help="CSV path (default <root>/manifest.csv)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    table = aggregate(root)
    out = Path(args.out) if args.out else root / "manifest.csv"
    if table.empty:
        print(f"no run directories with metadata.json under {root}")
        return
    table.to_csv(out, index=False)
    complete = table[table["status"] == "complete"]
    with_energy = table.dropna(subset=["energy_kwh"])
    print(f"{len(table)} run(s) ({len(complete)} complete) written to {out}")
    print(f"wall-clock total {table['wall_clock_min'].fillna(0).sum() / 60:.2f} h; "
          f"energy {with_energy['energy_kwh'].sum():.4f} kWh total, CPU share {with_energy['cpu_energy_kwh'].sum():.4f} kWh, "
          f"{with_energy['emissions_kg'].sum() * 1000:.1f} g CO2e over {len(with_energy)} run(s) with an energy record")
    shown = ["condition", "seed", "trainer", "threads", "step_budget", "git_describe", "wall_clock_min", "agent_steps_per_s",
             "energy_kwh", "selected_checkpoint", "selected_iqm_return", "status", "degenerate"]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 3):
        print(table[[c for c in shown if c in table.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
