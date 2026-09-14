"""Grid-level tables that use no behavioural metric: the training table and the degeneracy flags.

    uv run python scripts/report_grid.py --section training         # from the run manifests
    uv run python scripts/report_grid.py --section flags            # dashboard.summary + flags per run
    uv run python scripts/report_grid.py --root runs/_smoke_post --section all --n-eval-seeds 4

``--section training``: one row per run from ``metadata.json``, ``manifest.json`` and
``selection.json``: wall-clock, agent-steps per second, selected checkpoint, its IQM training
return, and the plateau check (last five finite checkpoint returns within 5% of the best).

``--section flags``: for every run, ``hamlet.metrics.dashboard.summary`` over the run's
stochastic evaluation Parquets (``<run_dir>/eval/stochastic/<condition>/<ckpt>/``, written
by ``scripts/evaluate_grid.py``) and the pre-registered degeneracy flags from
``hamlet.metrics.dashboard.flags``, applied per evaluation episode with ``greedy_return`` =
GREEDY-CLOCK's mean episode return on the same evaluation seed (from
``runs/A-S1-N<N>/GREEDY-CLOCK/seed<k>_ep0.parquet``): fewer than three zone types per
median agent-day; more than 70% of an agent's ticks in one zone; mean policy entropy below
0.05 nats; mean episode return below GREEDY-CLOCK's. A run is marked flagged when any of
the four fires: few_zones, single_zone and low_entropy when they fire on at least half of the
evaluation episodes; below_greedy under either that reading or the plan's run-level reading
(IQM of the run's episode returns below GREEDY-CLOCK's IQM, also its own column). Flagged runs are
marked, never dropped. The per-run result is also written to ``<run_dir>/flags.json`` so
``scripts/aggregate_manifest.py`` can carry it.

Outputs: ``<root>/reports/grid_training.csv`` and ``<root>/reports/grid_flags.csv`` plus
markdown on stdout. Nothing here imports ``hamlet.metrics.routine``, ``specialisation`` or
``social``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hamlet.config import HamletConfig  # noqa: E402
import analysis_constants as K  # noqa: E402

from hamlet.evaluate import evaluation_seeds  # noqa: E402

# Condition names carry the population size and follow the run root; set_population() rebinds
# them once --root is known, so an archived root at another N is read under its own names.
NN = HamletConfig().n_agents
A_S1 = f"A-S1-N{NN}"


def set_population(root) -> None:
    global NN, A_S1
    NN = K.root_n_agents(root)
    A_S1 = f"A-S1-N{NN}"
from hamlet.metrics.dashboard import flags, mean_return, summary, traits_file_for  # noqa: E402
from hamlet.metrics.stats import iqm  # noqa: E402

PLATEAU_TOLERANCE = 0.05
FLAGGED_EPISODE_FRACTION = 0.5
STOCHASTIC_PASS = "stochastic"


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/seed*/selection.json"))


def training_row(run_dir: Path) -> dict[str, Any]:
    meta = _load(run_dir / "metadata.json")
    manifest = _load(run_dir / "manifest.json")
    selection = _load(run_dir / "selection.json")
    progress = pd.read_csv(run_dir / "progress.csv") if (run_dir / "progress.csv").exists() else pd.DataFrame()
    last = progress.iloc[-1] if len(progress) else {}
    wall = meta.get("wall_clock_s") or (float(last["elapsed_s"]) if len(progress) else np.nan)
    steps = meta.get("agent_steps") or (int(last["agent_steps"]) if len(progress) else np.nan)
    cands = selection.get("candidates", [])
    iqms = [c.get("iqm_return") for c in cands]
    finite = [v for v in iqms if v is not None and np.isfinite(v)]
    best = max(finite) if finite else np.nan
    # Over the last five *finite* returns: the final checkpoint carries no completed episodes, so
    # its IQM is NaN. The selection rule is untouched. Matches scripts/report_pilot.py.
    plateau = bool(finite) and all(abs(v - best) <= PLATEAU_TOLERANCE * abs(best) for v in finite)
    selected = selection.get("selected")
    sel_iqm = next((c.get("iqm_return") for c in cands if c.get("path") == selected), np.nan)
    git = meta.get("git") or manifest.get("git") or {}
    return {
        "condition": meta.get("condition", run_dir.parent.name), "seed": meta.get("env_seed", int(run_dir.name.removeprefix("seed"))),
        "trainer": meta.get("trainer"), "threads": meta.get("threads", manifest.get("threads")),
        "step_budget": meta.get("step_budget"), "agent_steps": steps, "updates": meta.get("updates", last.get("update") if len(progress) else None),
        "wall_clock_min": float(wall) / 60 if wall is not None else np.nan, "agent_steps_per_s": meta.get("agent_steps_per_s"),
        "selected_checkpoint": Path(selected).name if selected else None, "selected_iqm_return": sel_iqm,
        "last_five_iqm": iqms, "best_of_last_five": best,
        "plateau_spread_frac": (max(finite) - min(finite)) / abs(best) if finite and best else np.nan, "plateau": plateau,
        "git_describe": git.get("describe"), "status": meta.get("status"),
    }


def stochastic_parquets(run_dir: Path, condition: str) -> list[Path]:
    """The stochastic-pass Parquets of a run (one per evaluation seed)."""
    base = run_dir / "eval" / STOCHASTIC_PASS / condition
    files = sorted(base.glob("*/seed*_ep*.parquet"))
    return files


def flags_row(run_dir: Path, condition: str, greedy_root: Path, n_eval_seeds: int, burn_in_days: Optional[int]) -> Optional[dict[str, Any]]:
    files = stochastic_parquets(run_dir, condition)
    if not files:
        return None
    seeds = set(evaluation_seeds(n_eval_seeds))
    counts = {"few_zones": 0, "single_zone": 0, "low_entropy": 0, "below_greedy": 0, "degenerate": 0}
    ppo_returns, gc_returns, summaries, n, missing_gc = [], [], [], 0, 0
    for path in files:
        df = pd.read_parquet(path, engine="pyarrow")
        seed = int(df["seed"].iloc[0])
        if seed not in seeds:
            continue
        gc_path = greedy_root / A_S1 / "GREEDY-CLOCK" / f"seed{seed}_ep0.parquet"
        gc_ret = mean_return(pd.read_parquet(gc_path, engine="pyarrow")) if gc_path.exists() else None
        missing_gc += int(gc_ret is None)
        f = flags(df, greedy_return=gc_ret, burn_in_days=burn_in_days)
        for k in counts:
            counts[k] += int(bool(f[k]))
        ppo_returns.append(f["mean_return"])
        if gc_ret is not None:
            gc_returns.append(gc_ret)
        summaries.append(summary(df, burn_in_days, traits_file=traits_file_for(path)))
        n += 1
    if n == 0:
        return None
    table = pd.concat(summaries, ignore_index=True)
    numeric = table.select_dtypes(include="number").mean(numeric_only=True).to_dict()
    if missing_gc:
        print(f"warning: {condition} {run_dir.name}: GREEDY-CLOCK missing on {missing_gc} evaluation seed(s); below_greedy judged on the rest", flush=True)
    below_greedy_iqm = bool(gc_returns) and iqm(np.asarray(ppo_returns)) < iqm(np.asarray(gc_returns))
    half = FLAGGED_EPISODE_FRACTION * n
    per_flag = {"few_zones": counts["few_zones"] >= half, "single_zone": counts["single_zone"] >= half,
                "low_entropy": counts["low_entropy"] >= half,
                # the return component fires under either reading: the code's per-episode rule or the plan's IQM rule
                "below_greedy": (counts["below_greedy"] >= half) or below_greedy_iqm}
    degenerate = any(per_flag.values())
    row = {"condition": condition, "seed": int(run_dir.name.removeprefix("seed")), "n_episodes": n,
           **{f"flag_{k}_episodes": v for k, v in counts.items()},
           "flagged": degenerate, "ppo_return_iqm": iqm(np.asarray(ppo_returns)),
           "greedy_return_iqm": iqm(np.asarray(gc_returns)) if gc_returns else np.nan,
           "below_greedy_iqm": below_greedy_iqm, "greedy_missing_seeds": missing_gc,
           **{f"mean_{k}": v for k, v in numeric.items() if k not in ("seed", "episode", "n_agents")}}
    (run_dir / "flags.json").write_text(json.dumps({
        **per_flag, "degenerate": degenerate, "below_greedy_iqm": below_greedy_iqm, "n_episodes": n,
        "greedy_missing_seeds": missing_gc, "episode_counts": counts,
        "rule": f"few_zones, single_zone, low_entropy: fire on at least {FLAGGED_EPISODE_FRACTION:.0%} of the evaluation episodes; "
                "below_greedy: that reading or the plan's IQM-return reading; degenerate = any"}, indent=2))
    return row


def baseline_gain_table(greedy_root: Path) -> str:
    """GREEDY-CLOCK's and GREEDY-CLOCK-WORK's clock gain per evaluation seed from the 3-seed and 32-seed baseline summaries.

    ``baseline_summary.csv`` was produced by ``scripts/run_baselines.py --n-seeds 3`` (state-bin
    terciles pooled over those 3 episodes) and ``baseline_summary_n32.csv`` by ``--n-seeds 32``
    (terciles pooled over 32); the table shows the same seeds under both references, values only.
    """
    a = pd.read_csv(greedy_root / A_S1 / "baseline_summary.csv")
    b = pd.read_csv(greedy_root / A_S1 / "baseline_summary_n32.csv")
    rows = []
    for policy in ("GREEDY-CLOCK", "GREEDY-CLOCK-WORK"):
        pa = a[a.policy == policy].set_index("seed")
        pb = b[b.policy == policy].set_index("seed")
        for seed in sorted(pa.index):
            rows.append({"policy": policy, "seed": seed, "clock_gain_bits_n3": pa.loc[seed, "clock_gain_bits"],
                         "clock_gain_bits_n32": pb.loc[seed, "clock_gain_bits"], "clock_gain_z_n3": pa.loc[seed, "clock_gain_z"],
                         "clock_gain_z_n32": pb.loc[seed, "clock_gain_z"]})
        rows.append({"policy": policy, "seed": "mean over 32", "clock_gain_bits_n3": np.nan,
                     "clock_gain_bits_n32": pb["clock_gain_bits"].mean(), "clock_gain_z_n3": np.nan, "clock_gain_z_n32": pb["clock_gain_z"].mean()})
    return "## clock gain (terciles), n = 3 against n = 32 pooled reference\n" + pd.DataFrame(rows).round(3).to_markdown(index=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", choices=["all", "training", "flags", "baseline-gain"], default="all")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--greedy-root", default="runs", help="root holding A-S1-N<N>/GREEDY-CLOCK/seed*_ep0.parquet")
    parser.add_argument("--n-eval-seeds", type=int, default=32)
    parser.add_argument("--burn-in-days", type=int, default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    set_population(root)
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.section == "baseline-gain":
        print(baseline_gain_table(Path(args.greedy_root)))
        return
    dirs = run_dirs(root)
    if not dirs:
        print(f"no finished runs (selection.json) under {root}")
        return
    pd.set_option("display.width", 240)
    if args.section in ("all", "training"):
        table = pd.DataFrame([training_row(d) for d in dirs])
        table.to_csv(out_dir / "grid_training.csv", index=False)
        shown = table.drop(columns=["last_five_iqm"])
        print("## training\n" + shown.round(3).to_markdown(index=False) + "\n")
        print(f"plateau in {int(table['plateau'].sum())} of {len(table)} run(s); "
              f"mean wall-clock {table['wall_clock_min'].mean():.1f} min; mean rate {table['agent_steps_per_s'].dropna().mean():.0f} agent-steps/s\n")
    if args.section in ("all", "flags"):
        rows = []
        for d in dirs:
            cond = _load(d / "metadata.json").get("condition", d.parent.name)
            row = flags_row(d, cond, Path(args.greedy_root), args.n_eval_seeds, args.burn_in_days)
            if row is None:
                print(f"{cond} seed {d.name}: no stochastic evaluation Parquets under {d / 'eval' / STOCHASTIC_PASS}; run scripts/evaluate_grid.py first")
                continue
            rows.append(row)
        if rows:
            table = pd.DataFrame(rows)
            table.to_csv(out_dir / "grid_flags.csv", index=False)
            shown = [c for c in table.columns if c.startswith(("condition", "seed", "n_episodes", "flag_", "flagged", "ppo_return", "greedy_return", "below_greedy"))]
            print("## flags\n" + table[shown].round(3).to_markdown(index=False) + "\n")
            print(f"{int(table['flagged'].sum())} of {len(table)} run(s) flagged (marked, never dropped); written to {out_dir / 'grid_flags.csv'}")


if __name__ == "__main__":
    main()
