"""Gate 2 report for the pilot: training return, mean drive and the degeneracy flags, nothing else.

    uv run python scripts/report_pilot.py                       # runs/A-S1-N<N>/seed{0,1,2}, 32 evaluation seeds
    uv run python scripts/report_pilot.py --seeds 0 --n-eval-seeds 4 --root runs/_smoke_pilot   # smoke

Reads the pilot run directories written by ``scripts/run_grid.py`` and asserts nothing.
Per seed it prints wall-clock, agent-steps per second, the plateau check (the last five
checkpoints' IQM training returns within 5% of the best), the selected checkpoint and its
IQM return; it then evaluates the selected checkpoint on the evaluation seeds
(stochastic sampling, ``hamlet.evaluate.evaluate``, skipped where the Parquet already
exists) and prints mean drive over days 1-5 paired against GREEDY-CLOCK on the same
evaluation seeds, and ``hamlet.metrics.dashboard.flags`` per seed. The four Gate 2
criteria of the schedule are printed with PASS or FAIL and a verdict, followed by the
pre-listed fallback order, as text.

What this script deliberately does not do: it imports nothing from
``hamlet.metrics.routine``, ``hamlet.metrics.specialisation`` or ``hamlet.metrics.social``
and prints no routine, specialisation or co-presence number. Checkpoint selection is
read from ``selection.json`` (training return only) and never revisited here.

Flag semantics: ``dashboard.flags`` is applied per evaluation episode with
``greedy_return`` set to GREEDY-CLOCK's mean episode return on the same evaluation seed
(the function's own definitions: mean episode return, per-agent 70% share); a pilot seed
counts as flagged when the ``degenerate`` flag fires on at least half of its evaluation
episodes. The pre-registration's run-level reading of ``below_greedy`` (IQM of the
evaluation returns below GREEDY-CLOCK's IQM) is printed beside it.

GREEDY-CLOCK on all 32 evaluation seeds comes from ``scripts/run_baselines.py --n-seeds 32``
(the existing Parquets are read; the command is printed when any seed is missing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamlet.config import HamletConfig  # noqa: E402
from hamlet.evaluate import evaluate, evaluation_seeds, make_policy  # noqa: E402
from hamlet.train_common import PPOHyperparameters  # noqa: E402
from hamlet.metrics.common import after_burn_in  # noqa: E402
from hamlet.metrics.dashboard import flags, mean_return  # noqa: E402
from hamlet.metrics.stats import iqm  # noqa: E402

PLATEAU_TOLERANCE = 0.05
BUDGET_MIN_PER_RUN = (8.0, 15.0)     # compute-budget estimate for the fallback PPO over the batched core (superseded by measurement)
BUDGET_GRID_H = (1.6, 3.0)           # the same table's cell for the 48-run grid at 4 concurrent
BUDGET_TABLE_STEPS = 10_000_000      # the budget those two cells were written at; both scale with it
GRID_RUNS = 48
FLAGGED_EPISODE_FRACTION = 0.5
GATE_CRITERIA = (
    "PPO beats GREEDY-CLOCK on mean drive (days 1 to 5, 32 evaluation episodes, paired on evaluation seeds) in every evaluated pilot seed.",
    "At most 1 pilot seed flagged degenerate.",
    "Learning curve of training return has flattened by the step budget (last five checkpoints within 5% of the best).",
    "Wall-clock per run measured and the grid fits the budget table below.",
)
CRITERION_4_NOTE = ("criterion 4 is read as the schedule intends: the per-run wall-clock is measured at the pilot's concurrency and the "
                    "48-run grid is projected at the launch concurrency (ceil(48 / concurrent) waves x per-run time); it passes when "
                    "every pilot run has a measured time and the projection fits the deadline; the table's 8 to 15 min and 1.6 to 3 h "
                    "cells were estimates that the measurement replaces. A failure of criterion 4 alone calls for no constant "
                    "revision: it is answered by the grid tier (--grid minimum, --grid minimum-noclock-dropped) or the cluster.")
FALLBACK_TEXT = """If Gate 2 fails:

- PPO below GREEDY-CLOCK: exactly one constant revision, chosen from this pre-listed order, recorded against the pre-registration: (1) `gamma` 0.995 to 0.997; (2) `entropy_coeff` 0.01 to 0.003; (3) `market_open` widened; (4) `cap_canteen` to ceil(N/3). Re-pilot with 2 seeds. Launch the grid regardless of the re-pilot result, and the paper reports what happened.
- Curve not flat at 10M: the grid still launches at 10M (matched budget is what is compared) and the cluster request for 40M re-runs of A-S1 and D-S1 goes in the same day as the stretch.
- More than one pilot seed flagged: the flags are reported; if all three are flagged the world constants are wrong and revision (3) or (4) applies first."""


def training_summary(run_dir: Path) -> dict:
    """Wall-clock, rate, plateau and selection of one run from metadata.json, progress.csv and selection.json."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    progress = pd.read_csv(run_dir / "progress.csv")
    selection = json.loads((run_dir / "selection.json").read_text())
    last = progress.iloc[-1]
    wall = float(meta.get("wall_clock_s") or last["elapsed_s"])
    steps = int(meta.get("agent_steps") or last["agent_steps"])
    candidates = selection["candidates"]
    iqms = [c["iqm_return"] for c in candidates]
    finite = [v for v in iqms if v is not None and np.isfinite(v)]
    best = max(finite) if finite else float("nan")
    # Over the last five *finite* returns. The final checkpoint is written one update after the
    # previous one and episodes complete only every sixth, so it carries no finished episodes and
    # its IQM is NaN; requiring all five to be finite failed every run at a spread of under 1%.
    # The selection rule is untouched: it still reads the last five checkpoints.
    within = bool(finite) and all(abs(v - best) <= PLATEAU_TOLERANCE * abs(best) for v in finite)
    selected = Path(selection["selected"]) if selection.get("selected") else None
    sel_iqm = next((c["iqm_return"] for c in candidates if c["path"] == selection.get("selected")), float("nan"))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "run_dir": str(run_dir), "condition": meta["condition"], "seed": int(meta["env_seed"]),
        "wall_clock_min": wall / 60, "agent_steps": steps, "agent_steps_per_s": steps / max(wall, 1e-9),
        "updates": int(last["update"]), "step_budget": int(meta.get("step_budget") or meta["hyperparameters"]["total_agent_steps"]),
        "last_five_iqm": iqms, "best_iqm": best, "plateau": within,
        "plateau_spread_frac": (max(finite) - min(finite)) / abs(best) if finite and best else float("nan"),
        "selected": str(selected) if selected else None, "selected_iqm": sel_iqm,
        "git": meta.get("git"), "energy": meta.get("energy") or {}, "emissions_kg": meta.get("emissions_kg"),
        "hamlet_config": meta.get("hamlet_config"), "concurrent": manifest.get("concurrent"),
    }


def config_of_run(run: dict, fallback: HamletConfig) -> HamletConfig:
    """The HamletConfig the run was trained with (metadata.json), so evaluation happens in the same world."""
    raw = run.get("hamlet_config")
    if not raw:
        return fallback
    fields = {f for f in HamletConfig.__dataclass_fields__}
    cfg = HamletConfig(**{k: v for k, v in raw.items() if k in fields})
    if cfg.job_open is not None:
        cfg.job_open = tuple(cfg.job_open)
    for name in ("market_open", "night"):
        if getattr(cfg, name, None) is not None:
            setattr(cfg, name, tuple(getattr(cfg, name)))
    cfg.validate()
    return cfg


def greedy_is_fresh(cfg: HamletConfig, path: Path, seed: int) -> bool:
    """Re-roll GREEDY-CLOCK on one evaluation seed under ``cfg`` and compare its drive column with the stored file."""
    from hamlet.evaluate import rollout
    from hamlet.policies import GreedyClockPolicy

    stored = pd.read_parquet(path, engine="pyarrow")
    fresh = rollout(cfg, GreedyClockPolicy(), seed, 0)
    return len(stored) == len(fresh) and np.array_equal(stored["D"].to_numpy(), fresh["D"].to_numpy())


def evaluate_selected(run_dir: Path, selected: Path, cfg: HamletConfig, seeds: Sequence[int], overwrite: bool = False) -> list[Path]:
    """Stochastic evaluation of the selected checkpoint into ``<run_dir>/eval/stochastic/<condition>/<ckpt>/``.

    The same layout ``scripts/evaluate_grid.py`` uses for its stochastic pass, so the
    pilot's evaluation is the grid's primary pass for those three cells and is not redone.
    """
    policy, label = make_policy(fallback=str(selected))
    return evaluate(cfg, policy, seeds, run_dir / "eval" / "stochastic", checkpoint=label, overwrite=overwrite)


def mean_drive_days_1_to_5(df: pd.DataFrame, burn_in_days: int) -> float:
    return float(after_burn_in(df, burn_in_days)["D"].mean())


def baseline_paths(baseline_root: Path, condition: str, seeds: Sequence[int]) -> dict[int, Path]:
    return {s: baseline_root / condition / "GREEDY-CLOCK" / f"seed{s}_ep0.parquet" for s in seeds}


def seed_report(run: dict, paths: Sequence[Path], greedy: dict[int, Path], burn_in_days: int) -> dict:
    """Mean drive paired against GREEDY-CLOCK and the degeneracy flags for one pilot seed."""
    rows, flagged, per_flag = [], 0, {"few_zones": 0, "single_zone": 0, "low_entropy": 0, "below_greedy": 0}
    ppo_returns, gc_returns = [], []
    for path in paths:
        df = pd.read_parquet(path, engine="pyarrow")
        seed = int(df["seed"].iloc[0])
        gc = pd.read_parquet(greedy[seed], engine="pyarrow")
        gc_ret = mean_return(gc)
        f = flags(df, greedy_return=gc_ret, burn_in_days=burn_in_days)
        flagged += int(bool(f["degenerate"]))
        for k in per_flag:
            per_flag[k] += int(bool(f[k]))
        ppo_returns.append(f["mean_return"])
        gc_returns.append(gc_ret)
        rows.append({"eval_seed": seed, "ppo_mean_D": mean_drive_days_1_to_5(df, burn_in_days),
                     "greedy_mean_D": mean_drive_days_1_to_5(gc, burn_in_days),
                     "ppo_return": f["mean_return"], "greedy_return": gc_ret,
                     "entropy_nats": f["mean_entropy_nats"], "max_zone_share": f["max_zone_share"],
                     "median_distinct_zones": f["median_distinct_zones"], "degenerate": bool(f["degenerate"])})
    table = pd.DataFrame(rows)
    diff = table["ppo_mean_D"] - table["greedy_mean_D"]
    n = len(table)
    return {
        "seed": run["seed"], "n_eval": n, "table": table,
        "ppo_mean_D": float(table["ppo_mean_D"].mean()), "greedy_mean_D": float(table["greedy_mean_D"].mean()),
        "paired_diff_mean": float(diff.mean()), "paired_diff_iqm": iqm(diff.to_numpy()),
        "frac_eval_seeds_ppo_lower": float((diff < 0).mean()),
        "beats_greedy": bool(diff.mean() < 0),
        "episodes_flagged": flagged, "flag_counts": per_flag,
        "flagged": bool(n and flagged >= FLAGGED_EPISODE_FRACTION * n),
        "ppo_return_iqm": iqm(np.asarray(ppo_returns)), "greedy_return_iqm": iqm(np.asarray(gc_returns)),
        "below_greedy_iqm": bool(iqm(np.asarray(ppo_returns)) < iqm(np.asarray(gc_returns))),
    }


def energy_line(runs: Sequence[dict]) -> Optional[str]:
    """Per-run energy from each run's own record (metadata.json ``energy``, written from the tracker), not the shared CSV."""
    parts = []
    for r in runs:
        e = r.get("energy") or {}
        if not e or e.get("cpu_energy_kwh") is None:
            continue
        cpu_wh = 1000 * float(e["cpu_energy_kwh"])
        total_wh = 1000 * float(e.get("energy_kwh") or 0)
        parts.append(f"seed {r['seed']}: CPU {cpu_wh:.2f} Wh (process share), total incl. RAM constant "
                     f"{total_wh:.2f} Wh, {1000 * float(e.get('emissions_kg') or 0):.2f} g CO2e, "
                     f"{float(e.get('duration_s') or 0) / 60:.1f} min")
    return "energy (codecarbon, per-run records): " + "; ".join(parts) if parts else None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs", help="run root written by run_grid.py")
    parser.add_argument("--condition", default=HamletConfig().condition_name)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="pilot training seeds")
    parser.add_argument("--n-eval-seeds", type=int, default=32, help="evaluation seeds 10000.. (32 for the gate)")
    parser.add_argument("--baseline-root", default="runs", help="root holding <condition>/GREEDY-CLOCK/seed*_ep0.parquet")
    parser.add_argument("--no-eval", action="store_true", help="only the training-side rows")
    parser.add_argument("--overwrite-eval", action="store_true")
    parser.add_argument("--grid-concurrent", type=int, default=4, help="concurrency the grid will launch with (criterion 4 projection)")
    parser.add_argument("--grid-deadline-h", type=float, default=12.0, help="hours the grid must fit in (criterion 4: a compute budget, not a calendar)")
    args = parser.parse_args(argv)

    root, baseline_root = Path(args.root), Path(args.baseline_root)
    cfg = HamletConfig()
    if cfg.condition_name != args.condition:
        parts = args.condition.split("-")
        cfg = HamletConfig(arm=parts[0], symmetry=parts[1], clock_visible="noclock" not in parts)
        cfg.validate()
    burn_in = cfg.burn_in_days
    eval_seeds = evaluation_seeds(args.n_eval_seeds)
    print(f"Gate 2 report: {args.condition}, pilot seeds {args.seeds}, {len(eval_seeds)} evaluation seeds, root {root}")
    print("uses training return, mean drive (days 1-5) and dashboard.flags only; no behavioural metric is computed\n")

    runs, missing = [], []
    for s in args.seeds:
        run_dir = root / args.condition / f"seed{s}"
        if not (run_dir / "selection.json").exists():
            missing.append(s)
            continue
        runs.append(training_summary(run_dir))
    if missing:
        print(f"incomplete pilot seed(s) {missing}: no selection.json under {root / args.condition}; nothing evaluated for them\n")

    print("== training side")
    for r in runs:
        five = ", ".join("nan" if v is None or not np.isfinite(v) else f"{v:.1f}" for v in r["last_five_iqm"])
        print(f"seed {r['seed']}: wall-clock {r['wall_clock_min']:.1f} min, {r['agent_steps_per_s']:.0f} agent-steps/s, "
              f"{r['agent_steps']} steps in {r['updates']} updates (budget {r['step_budget']}); "
              f"last five IQM returns [{five}] best {r['best_iqm']:.1f} spread {100 * r['plateau_spread_frac']:.1f}% "
              f"-> plateau {'yes' if r['plateau'] else 'no'}; selected {Path(r['selected']).name if r['selected'] else None} "
              f"IQM {r['selected_iqm']:.1f}; git {r['git']['describe'] if r.get('git') else 'unknown'}")
    line = energy_line(runs)
    if line:
        print(line)
    dirty = [r["seed"] for r in runs if (r.get("git") or {}).get("describe", "").endswith("-dirty")]
    if dirty:
        print(f"WARNING: seed(s) {dirty} were trained from a dirty working tree (git describe ends in -dirty); "
              "the pre-registration requires a tagged tree for confirmatory runs")

    reports = []
    if not args.no_eval and runs:
        greedy = baseline_paths(baseline_root, args.condition, eval_seeds)
        absent = [s for s, p in greedy.items() if not p.exists()]
        if absent:
            print(f"\nGREEDY-CLOCK is missing on evaluation seed(s) {absent[:5]}{'...' if len(absent) > 5 else ''}; regenerate with:\n"
                  f"  uv run python scripts/run_baselines.py --n-seeds {args.n_eval_seeds}\nthen rerun this report.")
            raise SystemExit(2)
        run_cfg = config_of_run(runs[0], cfg)
        if not greedy_is_fresh(run_cfg, greedy[eval_seeds[0]], eval_seeds[0]):
            print(f"\nGREEDY-CLOCK Parquets under {baseline_root} were not produced by the world these runs trained in "
                  f"(a constant revision since the baselines were rolled); regenerate with:\n"
                  f"  uv run python scripts/run_baselines.py --n-seeds {args.n_eval_seeds} --overwrite\nthen rerun this report.")
            raise SystemExit(2)
        print("\n== evaluation side (stochastic sampling, selected checkpoint, days 1-5, world = the run's own config)")
        for r in runs:
            t0 = time.perf_counter()
            paths = evaluate_selected(Path(r["run_dir"]), Path(r["selected"]), config_of_run(r, cfg), eval_seeds, overwrite=args.overwrite_eval)
            rep = seed_report(r, paths, greedy, burn_in)
            reports.append(rep)
            print(f"seed {rep['seed']}: PPO mean drive {rep['ppo_mean_D']:.3f} vs GREEDY-CLOCK {rep['greedy_mean_D']:.3f}; "
                  f"paired difference mean {rep['paired_diff_mean']:+.3f} (IQM {rep['paired_diff_iqm']:+.3f}), PPO lower on "
                  f"{100 * rep['frac_eval_seeds_ppo_lower']:.0f}% of {rep['n_eval']} evaluation seeds -> "
                  f"{'beats' if rep['beats_greedy'] else 'does not beat'} GREEDY-CLOCK; "
                  f"flags: degenerate on {rep['episodes_flagged']}/{rep['n_eval']} episodes "
                  f"(few_zones {rep['flag_counts']['few_zones']}, single_zone {rep['flag_counts']['single_zone']}, "
                  f"low_entropy {rep['flag_counts']['low_entropy']}, below_greedy {rep['flag_counts']['below_greedy']}) -> "
                  f"{'FLAGGED' if rep['flagged'] else 'not flagged'}; IQM return {rep['ppo_return_iqm']:.1f} vs GREEDY-CLOCK "
                  f"{rep['greedy_return_iqm']:.1f} ({'below' if rep['below_greedy_iqm'] else 'not below'}, plan reading); "
                  f"{time.perf_counter() - t0:.0f} s")

    print("\n== Gate 2 criteria")
    # The criteria read the cells that are actually present, so a two-seed pilot is judged as a
    # two-seed pilot: every evaluated cell must beat GREEDY-CLOCK, at most one may be flagged, and
    # every finished cell must have plateaued.
    n_pilot = len(reports) if reports else len(args.seeds)
    c1 = bool(reports) and all(rep["beats_greedy"] for rep in reports)
    c2 = bool(reports) and sum(1 for rep in reports if rep["flagged"]) <= 1
    c3 = bool(runs) and all(r["plateau"] for r in runs)
    walls = ", ".join("%.1f" % r["wall_clock_min"] for r in runs)
    per_run_min = max((r["wall_clock_min"] for r in runs), default=float("nan"))
    waves = -(-GRID_RUNS // args.grid_concurrent)
    grid_h = waves * per_run_min / 60 if runs else float("nan")
    c4 = bool(runs) and np.isfinite(grid_h) and grid_h <= args.grid_deadline_h
    budget = PPOHyperparameters().total_agent_steps
    scale = budget / BUDGET_TABLE_STEPS          # the budget estimates were written at 10M
    est_run = (BUDGET_MIN_PER_RUN[0] * scale, BUDGET_MIN_PER_RUN[1] * scale)
    est_grid = (BUDGET_GRID_H[0] * scale, BUDGET_GRID_H[1] * scale)
    details = [
        f"{sum(1 for rep in reports if rep['beats_greedy'])} of {len(reports)} evaluated seed(s) beat GREEDY-CLOCK "
        f"(every one must)",
        f"{sum(1 for rep in reports if rep['flagged'])} of {len(reports)} evaluated seed(s) flagged",
        f"plateau in {sum(1 for r in runs if r['plateau'])} of {len(runs)} finished seed(s)",
        f"per-run wall-clock {walls} min measured at the pilot's concurrency; {GRID_RUNS}-run grid projected at "
        f"{args.grid_concurrent} concurrent = {waves} waves x {per_run_min:.1f} min = {grid_h:.1f} h against the {args.grid_deadline_h:.0f} h deadline "
        f"at the {budget / 1e6:.0f}M budget (table estimates scaled from 10M: {est_run[0]:.0f} to {est_run[1]:.0f} min "
        f"per run, {est_grid[0]:.1f} to {est_grid[1]:.1f} h for the grid)",
    ]
    for ok, text, detail in zip((c1, c2, c3, c4), GATE_CRITERIA, details):
        print(f"[{'PASS' if ok else 'FAIL'}] {text}  ({detail})")
    print(CRITERION_4_NOTE)
    if not reports and not args.no_eval:
        print("(criteria 1 and 2 need the evaluation side; nothing was evaluated)")
    failed = [name for ok, name in zip((c1, c2, c3, c4), ("1 drive", "2 flags", "3 plateau", "4 wall-clock")) if not ok]
    verdict = "PASS" if not failed else "FAIL"
    if verdict == "PASS":
        print("\nVERDICT: Gate 2 PASS")
    elif failed == ["4 wall-clock"]:
        print("\nVERDICT: Gate 2 FAIL on criterion 4 only: no constant revision; choose the grid tier or the cluster (see the fallback text)")
    else:
        print(f"\nVERDICT: Gate 2 FAIL on criterion(s) {', '.join(failed)}; one constant revision from the pre-listed order, recorded against the pre-registration")
    print("\n" + FALLBACK_TEXT)


if __name__ == "__main__":
    main()
