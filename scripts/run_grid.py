"""Launch the pilot and the confirmatory training grid: seed-major cells, concurrent runs, manifests, resume.

Grids (``--grid``), all on the neutral population and eight paired seeds unless said:

  confirmatory              48 runs: A-S1, D-S1, A-S0, A-S1-noclock, B-S1, C-S1 x seeds 0..7 (default)
  minimum                   32 runs: the first four conditions (the fall-back that loses the 2x2)
  minimum-noclock-dropped   24 runs: A-S1, D-S1, A-S0 (the fall-back that also loses the clock contrast)
  full                      60 runs: all six conditions x seeds 0..9 (the ten-seed stretch)

Order (``--order``): ``minimum-first`` (default) runs the four minimum conditions for
every seed first and then B-S1 and C-S1, the order the pre-registration's section 3
registers, so that a grid that slips still completes the 32-run minimum first and the
fall-back ladder (48 to 32 to 24) keeps its meaning; ``interleaved`` runs every condition
of a seed before the next seed, so a partial grid is balanced across all six conditions
at every point (this departs from the registered order).

The pilot is a subset of the same grid, into the same run directories::

    uv run python scripts/run_grid.py --conditions A-S1 --seeds 0 1 2 --concurrent 3 --threads 2

so three passed pilot runs are three finished grid cells and the full launch skips them.

Per cell the launcher writes ``<out>/<condition>/seed<s>/manifest.json`` (config JSON, seed,
git describe and commit, hyperparameters, step budget, command, threads, start and end
timestamps, exit code, status) and ``train.log`` (the trainer's stdout); the trainer
writes ``metadata.json``, ``progress.csv``, the checkpoints and, last, ``selection.json``.
A cell is complete when ``selection.json`` exists and its recorded step budget equals the
launch budget; it is then skipped on relaunch. A finished cell at another budget (a smoke
run in the study root) makes the launcher refuse rather than skip or overwrite; a cell whose
trainer died (no ``selection.json``) is run again from scratch. Study launches (the default
``runs`` root without ``--agent-steps``) refuse to start from a dirty or untagged working
tree, because every manifest records ``git describe`` and the pre-registration requires a
tagged tree; ``--agent-steps`` runs are smoke runs: they are refused in the study root and
do not log energy. Concurrency:
``--concurrent K`` trainers at once, each with ``--threads T`` torch threads and
``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = T`` in its environment; ``K * T`` must not exceed
the machine's logical CPUs. A guard refuses to start while any trainer (``hamlet.train_fallback``,
``sandbox.train``) is running or another launcher holds ``<out>/.run_grid.lock``
(``--force`` overrides). ``--estimate`` prints the wall-clock and disk estimate for the
selected cells from the measured agent-step rate and launches nothing; ``--dry-run``
prints the commands; ``--shell-list FILE`` writes them one per line.

The grid-level record ``<out>/grid_manifest.json`` lists every planned cell and its status
after each launcher invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamlet.config import HamletConfig  # noqa: E402
from hamlet.train_common import PPOHyperparameters, git_provenance, run_directory, utc_now  # noqa: E402
from hamlet.traits import POPULATIONS  # noqa: E402

TRAINER = "hamlet.train_fallback"
MINIMUM_SEEDS = tuple(range(0, 8))
EXTRA_SEEDS = tuple(range(8, 10))
ORDERS = ("minimum-first", "interleaved")
DEFAULT_ORDER = "minimum-first"
TRAINER_PATTERNS = ("hamlet.train_fallback", "sandbox.train")
LOCK_NAME = ".run_grid.lock"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# Measured with the study trainer at n_envs 16, 2 threads, alone on the machine
# (uv run python -m hamlet.train_fallback --seed 0 --n-envs 16 --threads 2 --agent-steps 122880 --out runs/_rate_smoke):
# 122,880 agent-steps in 9.2 s of loop time. --estimate uses finished runs under --out when any exist.
DEFAULT_RATE_AGENT_STEPS_PER_S = 13_000.0
CHECKPOINT_BYTES = 199_267          # one fallback checkpoint (2 x 128 MLP), measured on the smoke run
EVAL_PARQUET_BYTES = 687_717        # one evaluation-episode Parquet of a trained policy (log-probs present), measured
N_EVAL_SEEDS = 32


@dataclass(frozen=True)
class Condition:
    """One cell type of the grid and the command-line flags that select it."""

    arm: str
    symmetry: str
    clock_visible: bool = True
    population: str = "neutral"

    @property
    def config(self) -> HamletConfig:
        return HamletConfig(arm=self.arm, symmetry=self.symmetry, clock_visible=self.clock_visible, population=self.population)

    @property
    def name(self) -> str:
        return self.config.condition_name

    @property
    def short(self) -> str:
        """``A-S1``, ``A-S1-noclock``: the name without the ``-N8`` and population parts."""
        return f"{self.arm}-{self.symmetry}" + ("" if self.clock_visible else "-noclock")

    @property
    def flags(self) -> list[str]:
        out = ["--arm", self.arm, "--symmetry", self.symmetry]
        if not self.clock_visible:
            out.append("--noclock")
        if self.population != "neutral":
            out += ["--population", self.population]
        return out


CONDITIONS = (
    Condition("A", "S1"),
    Condition("D", "S1"),
    Condition("A", "S0"),
    Condition("A", "S1", clock_visible=False),
    Condition("B", "S1"),
    Condition("C", "S1"),
)
MINIMUM_CONDITIONS = CONDITIONS[:4]
FACTORIAL_CONDITIONS = CONDITIONS[4:]
GRIDS = {
    "confirmatory": (CONDITIONS, MINIMUM_SEEDS),
    "minimum": (MINIMUM_CONDITIONS, MINIMUM_SEEDS),
    "minimum-noclock-dropped": (CONDITIONS[:3], MINIMUM_SEEDS),
    "full": (CONDITIONS, MINIMUM_SEEDS + EXTRA_SEEDS),
}


def resolve_condition(text: str, population: str = "neutral") -> Condition:
    """``A-S1``, ``A-S1-noclock``, ``A-S1-N8`` or ``A-S1-N8-noclock`` to a grid condition."""
    for cond in CONDITIONS:
        c = replace(cond, population=population)
        if text in (c.short, c.name):
            return c
    raise ValueError(f"unknown condition {text!r}; choose from {[c.short for c in CONDITIONS]}")


def plan(
    grid: str = "confirmatory",
    order: str = DEFAULT_ORDER,
    conditions: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    population: str = "neutral",
) -> list[tuple[Condition, int]]:
    """``(condition, seed)`` cells in launch order, optionally restricted to ``conditions`` and ``seeds``."""
    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}; choose from {list(GRIDS)}")
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; choose from {ORDERS}")
    conds, grid_seeds = GRIDS[grid]
    conds = tuple(replace(c, population=population) for c in conds)
    if conditions:
        wanted = {resolve_condition(t, population).name for t in conditions}
        conds = tuple(c for c in conds if c.name in wanted)
        missing = wanted - {c.name for c in conds}
        if missing:
            raise ValueError(f"condition(s) {sorted(missing)} are not in grid {grid!r}")
    use_seeds = list(grid_seeds) if seeds is None else [int(s) for s in seeds]
    outside = [s for s in use_seeds if s not in grid_seeds]
    if outside:
        raise ValueError(f"seed(s) {outside} are outside grid {grid!r} (seeds {list(grid_seeds)})")
    if order == "interleaved":
        return [(c, s) for s in use_seeds for c in conds]
    first = [c for c in conds if c in tuple(replace(m, population=population) for m in MINIMUM_CONDITIONS)]
    rest = [c for c in conds if c not in first]
    return [(c, s) for s in use_seeds for c in first] + [(c, s) for s in use_seeds for c in rest]


def command(cond: Condition, seed: int, out: Path, threads: int,
            agent_steps: Optional[int], extra: Sequence[str]) -> list[str]:
    cmd = [sys.executable, "-m", TRAINER, *cond.flags, "--seed", str(seed), "--out", str(out),
           "--threads", str(threads)]
    if agent_steps is not None:
        cmd += ["--agent-steps", str(agent_steps)]
    return cmd + list(extra)


def running_trainers(patterns: Sequence[str] = TRAINER_PATTERNS) -> list[str]:
    """Command lines of live processes that look like a trainer (Linux ``/proc`` scan)."""
    hits = []
    me = os.getpid()
    for pid_dir in Path("/proc").glob("[0-9]*"):
        pid = int(pid_dir.name)
        if pid == me:
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if any(p in cmdline for p in patterns) and "run_grid.py" not in cmdline:
            hits.append(f"{pid}: {cmdline.strip()[:120]}")
    return hits


def pid_alive(pid: int) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def pid_is_trainer_of(pid: int, seed: int) -> bool:
    """True when ``pid`` is alive and its command line is a trainer for ``--seed <seed>`` (guards PID reuse)."""
    if not pid_alive(pid):
        return False
    try:
        cmdline = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return False
    return "hamlet.train_" in cmdline and f"--seed {seed} " in cmdline + " "


def acquire_lock(out: Path, force: bool) -> Path:
    """Refuse to start beside a live launcher on the same output root (``--force`` overrides)."""
    lock = out / LOCK_NAME
    if lock.exists() and not force:
        try:
            other = int(lock.read_text().strip())
        except ValueError:
            other = -1
        if pid_alive(other):
            raise SystemExit(f"refused: another launcher (pid {other}) holds {lock}; use --force to override")
    out.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    return lock


def recorded_budget(run_dir: Path) -> Optional[int]:
    """The step budget a finished run was launched with (metadata.json, else manifest.json), or None."""
    for name in ("metadata.json", "manifest.json"):
        path = run_dir / name
        if path.exists():
            try:
                m = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            budget = m.get("step_budget") or (m.get("hyperparameters") or {}).get("total_agent_steps")
            if budget:
                return int(budget)
    return None


def settings_hash(cfg: HamletConfig, hyper: PPOHyperparameters) -> str:
    """A short digest of the world config and the trainer settings a cell was produced with.

    Resume compares this, not just the presence of selection.json: a finished cell whose world or
    hyperparameters differ from the launch is a different experiment and must not be skipped. It is
    what stops a run finished under an earlier world or trainer being reused as a grid cell.
    """
    payload = {"hamlet_config": json.loads(cfg.to_json()), "hyperparameters": asdict(hyper)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def recorded_settings_hash(run_dir: Path) -> Optional[str]:
    """The settings digest of a finished run, recomputed from its own metadata.json."""
    path = run_dir / "metadata.json"
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    cfg_dict, hyper_dict = m.get("hamlet_config"), m.get("hyperparameters")
    if not cfg_dict or not hyper_dict:
        return None
    payload = {"hamlet_config": cfg_dict, "hyperparameters": hyper_dict}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cell_status(run_dir: Path, budget: Optional[int] = None, seed: Optional[int] = None,
                want_hash: Optional[str] = None) -> str:
    """``complete`` (selection.json, the launch budget and the launch settings), ``budget-mismatch``
    (finished at another budget), ``settings-mismatch`` (finished under a different world or trainer),
    ``running`` (manifest says so and its pid is that trainer), ``crashed`` or ``pending``."""
    if (run_dir / "selection.json").exists():
        done = recorded_budget(run_dir)
        if budget is not None and done is not None and int(done) != int(budget):
            return "budget-mismatch"
        if want_hash is not None:
            got = recorded_settings_hash(run_dir)
            if got is not None and got != want_hash:
                return "settings-mismatch"
        return "complete"
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            return "crashed"
        pid = m.get("pid", -1)
        own_seed = int(m.get("seed", seed if seed is not None else -1))
        if m.get("status") == "running" and pid_is_trainer_of(pid, own_seed):
            return "running"
        return "crashed"
    return "pending"


def write_manifest(run_dir: Path, cond: Condition, seed: int, cmd: Sequence[str], threads: int,
                   agent_steps: Optional[int], pid: int) -> Path:
    hyper = PPOHyperparameters() if agent_steps is None else PPOHyperparameters(total_agent_steps=int(agent_steps))
    cfg = cond.config
    manifest = {
        "condition": cond.name,
        "config": json.loads(cfg.to_json()),
        "seed": int(seed),
        "policy_seed": 100 + int(seed),
        "git": git_provenance(),
        "trainer": "fallback",
        "hyperparameters": hyper.__dict__,
        "step_budget": int(hyper.total_agent_steps),
        "threads": int(threads),
        "thread_env": {k: str(threads) for k in THREAD_ENV},
        "command": list(cmd),
        "pid": int(pid),
        "started_utc": utc_now(),
        "status": "running",
    }
    path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def finish_manifest(run_dir: Path, exit_code: int, started: float) -> dict:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    complete = (run_dir / "selection.json").exists()
    manifest.update({
        "finished_utc": utc_now(),
        "wall_clock_s": time.perf_counter() - started,
        "exit_code": int(exit_code),
        "status": "complete" if (exit_code == 0 and complete) else "failed",
        "selection_json": complete,
    })
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def write_grid_manifest(out: Path, grid: str, order: str, cells: Sequence[tuple[Condition, int]], budget: int,
                        hyper: Optional[PPOHyperparameters] = None) -> Path:
    path = out / "grid_manifest.json"
    record = {
        "grid": grid, "order": order, "written_utc": utc_now(), "git": git_provenance(),
        "cells": [{"condition": c.name, "seed": s, "run_dir": str(run_directory(out, c.config, s)),
                   "status": cell_status(run_directory(out, c.config, s), budget, s,
                                         settings_hash(c.config, hyper) if hyper is not None else None)}
                  for c, s in cells],
        "step_budget": budget,
    }
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def measured_rate(out: Path) -> Optional[float]:
    """Agent-steps per second from the finished runs under ``out`` (metadata ``agent_steps_per_s``), or None."""
    rates = []
    for meta in out.glob("*/seed*/metadata.json"):
        try:
            m = json.loads(meta.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("status") == "complete" and m.get("agent_steps_per_s"):
            rates.append(float(m["agent_steps_per_s"]))
    return sum(rates) / len(rates) if rates else None


def estimate(cells: Sequence[tuple[Condition, int]], concurrent: int, rate: float, agent_steps: int) -> dict:
    """Wall-clock (sequential and at ``concurrent``) and disk for ``cells`` at ``rate`` agent-steps per second per run."""
    n = len(cells)
    per_run_s = agent_steps / rate
    hyper = PPOHyperparameters(total_agent_steps=agent_steps)
    updates = -(-agent_steps // hyper.train_batch_size)
    checkpoints = updates // hyper.checkpoint_every_updates + (1 if updates % hyper.checkpoint_every_updates else 0)
    train_bytes = checkpoints * CHECKPOINT_BYTES + 400_000            # checkpoints plus logs, manifests, traits
    eval_bytes = N_EVAL_SEEDS * EVAL_PARQUET_BYTES
    return {
        "runs": n, "rate_agent_steps_per_s": rate, "per_run_min": per_run_s / 60,
        "sequential_h": n * per_run_s / 3600, "concurrent": concurrent,
        "concurrent_h": -(-n // concurrent) * per_run_s / 3600,
        "checkpoints_per_run": checkpoints, "disk_train_mb": n * train_bytes / 1e6,
        "disk_with_eval_mb": n * (train_bytes + eval_bytes) / 1e6,
    }


def launch(cells: Sequence[tuple[Condition, int]], out: Path, concurrent: int, threads: int,
           agent_steps: Optional[int], extra: Sequence[str]) -> list[dict]:
    """Run the cells ``concurrent`` at a time, skipping complete and running ones; returns the manifests."""
    env = dict(os.environ)
    env.update({k: str(threads) for k in THREAD_ENV})
    env["PYTHONUNBUFFERED"] = "1"                 # train.log keeps up with the trainer
    budget = int(agent_steps) if agent_steps is not None else PPOHyperparameters().total_agent_steps
    hyper = PPOHyperparameters() if agent_steps is None else replace(PPOHyperparameters(), total_agent_steps=budget)
    queue = list(cells)
    active: list[tuple[subprocess.Popen, Path, float, Condition, int, object]] = []
    results: list[dict] = []
    skipped_running = 0
    try:
        while queue or active:
            while queue and len(active) < concurrent:
                cond, seed = queue.pop(0)
                run_dir = run_directory(out, cond.config, seed)
                status = cell_status(run_dir, budget, seed, settings_hash(cond.config, hyper))
                if status == "budget-mismatch":
                    raise SystemExit(f"refused: {run_dir} is finished at budget {recorded_budget(run_dir)}, not {budget}; "
                                     "a smoke run sits in the study root; move or delete it before launching")
                if status == "settings-mismatch":
                    raise SystemExit(
                        f"refused: {run_dir} is finished under different settings "
                        f"(its own metadata hashes to {recorded_settings_hash(run_dir)}, this launch to "
                        f"{settings_hash(cond.config, hyper)}); it is a different experiment, so it is neither "
                        "skipped nor overwritten. Move or delete it before launching")
                if status in ("complete", "running"):
                    skipped_running += int(status == "running")
                    print(f"skip {cond.name} seed {seed}: {status}", flush=True)
                    continue
                if status == "crashed":
                    print(f"rerun {cond.name} seed {seed}: earlier attempt left no selection.json", flush=True)
                cmd = command(cond, seed, out, threads, agent_steps, extra)
                run_dir.mkdir(parents=True, exist_ok=True)
                log = open(run_dir / "train.log", "w")
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                write_manifest(run_dir, cond, seed, cmd, threads, agent_steps, proc.pid)
                print(f"run  {cond.name} seed {seed} (pid {proc.pid}): {' '.join(cmd)}", flush=True)
                active.append((proc, run_dir, time.perf_counter(), cond, seed, log))
            still = []
            for proc, run_dir, started, cond, seed, log in active:
                code = proc.poll()
                if code is None:
                    still.append((proc, run_dir, started, cond, seed, log))
                    continue
                log.close()
                manifest = finish_manifest(run_dir, code, started)
                results.append(manifest)
                print(f"{'done' if manifest['status'] == 'complete' else 'FAILED'} {cond.name} seed {seed} "
                      f"in {manifest['wall_clock_s'] / 60:.1f} min (exit {code})", flush=True)
            active = still
            if active:
                time.sleep(2.0)
    except KeyboardInterrupt:
        print("interrupted: stopping the running trainers", flush=True)
        for proc, run_dir, started, cond, seed, log in active:
            proc.terminate()
        for proc, run_dir, started, cond, seed, log in active:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
            manifest = finish_manifest(run_dir, proc.returncode if proc.returncode is not None else -1, started)
            manifest["status"] = "interrupted"
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            results.append(manifest)
        raise
    if skipped_running:
        print(f"{skipped_running} cell(s) skipped because another launcher is running them", flush=True)
    return results


def parse_seeds(text: Optional[str]) -> Optional[list[int]]:
    """``'0-7'`` or ``'0,3,5'`` or ``'0 1 2'`` to a list; ``None`` keeps the grid's own seeds."""
    if text is None:
        return None
    parts = []
    for chunk in text.replace(",", " ").split():
        if "-" in chunk:
            a, b = chunk.split("-")
            parts.extend(range(int(a), int(b) + 1))
        else:
            parts.append(int(chunk))
    return parts


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default="confirmatory", choices=sorted(GRIDS),
                        help="confirmatory 48 (default); minimum 32; minimum-noclock-dropped 24; full 60")
    parser.add_argument("--order", default=DEFAULT_ORDER, choices=ORDERS,
                        help="minimum-first: the registered order (default); interleaved: every condition of a seed before the next seed")
    parser.add_argument("--conditions", nargs="+", default=None, help="subset, e.g. A-S1 A-S0 or A-S1-N8-noclock")
    parser.add_argument("--seeds", nargs="+", default=None, help="subset, e.g. 0 1 2 or 0-7")
    parser.add_argument("--concurrent", type=int, default=4, help="trainers at once (default 4)")
    parser.add_argument("--threads", type=int, default=2, help="torch and BLAS threads per trainer (default 2)")
    parser.add_argument("--agent-steps", type=int, default=None,
                        help="override the registered step budget from PPOHyperparameters (smoke runs only)")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--population", default="neutral", choices=sorted(POPULATIONS), help="trait variant of every cell")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("--estimate", action="store_true", help="print wall-clock and disk estimates only")
    parser.add_argument("--rate", type=float, default=None, help="agent-steps per second per run for --estimate (default: measured runs, else the smoke figure)")
    parser.add_argument("--shell-list", default=None, help="write the commands to this file, one per line")
    parser.add_argument("--force", action="store_true", help="ignore a live trainer, lock, dirty tree or smoke-in-study-root refusal (never for a study launch)")
    parser.add_argument("extra", nargs="*", help="extra flags passed to every trainer call (after --)")
    args = parser.parse_args(argv)

    seeds = parse_seeds(" ".join(args.seeds)) if args.seeds else None
    cells = plan(args.grid, args.order, args.conditions, seeds, args.population)
    out = Path(args.out)
    cpus = os.cpu_count() or 1
    if args.concurrent * args.threads > cpus:
        raise SystemExit(f"refused: {args.concurrent} trainers x {args.threads} threads exceeds the {cpus} logical CPUs")
    study_launch = args.agent_steps is None and out.resolve() == Path("runs").resolve()
    extra = list(args.extra)
    if args.agent_steps is not None:
        if out.resolve() == Path("runs").resolve() and not args.force:
            raise SystemExit("refused: a smoke budget (--agent-steps) must not write into the study root 'runs'; use --out runs/_smoke_<name>")
        if "--no-codecarbon" not in extra:
            extra.append("--no-codecarbon")           # smoke runs never write the study's energy log
    budget = int(args.agent_steps) if args.agent_steps is not None else PPOHyperparameters().total_agent_steps
    hyper = PPOHyperparameters() if args.agent_steps is None else replace(PPOHyperparameters(), total_agent_steps=budget)
    commands = [command(c, s, out, args.threads, args.agent_steps, extra) for c, s in cells]
    lines = [" ".join(cmd) for cmd in commands]

    if args.shell_list:
        Path(args.shell_list).write_text("\n".join(lines) + "\n")
        print(f"{len(lines)} commands written to {args.shell_list}")
        return
    if args.dry_run:
        for (c, s), line in zip(cells, lines):
            print(f"[{cell_status(run_directory(out, c.config, s), budget, s, settings_hash(c.config, hyper))}] {line}")
        return
    if args.estimate:
        rate = args.rate or measured_rate(out) or DEFAULT_RATE_AGENT_STEPS_PER_S
        source = "--rate" if args.rate else ("finished runs under --out" if measured_rate(out) else "the smoke figure in this script")
        est = estimate(cells, args.concurrent, rate, args.agent_steps or PPOHyperparameters().total_agent_steps)
        pending = sum(1 for c, s in cells
                      if cell_status(run_directory(out, c.config, s), budget, s, settings_hash(c.config, hyper)) != "complete")
        print(json.dumps({**est, "pending_runs": pending, "rate_source": source}, indent=2))
        return

    live = running_trainers()
    if live and not args.force:
        raise SystemExit("refused: a trainer is running:\n  " + "\n  ".join(live) + "\nuse --force only for a deliberate overlap")
    provenance = git_provenance()
    if study_launch and provenance["describe"].endswith("-dirty") and not args.force:
        raise SystemExit("refused: study runs must start from a committed, tagged tree (every manifest records git describe, "
                         f"currently {provenance['describe']!r}); commit and tag analysis-plan-v1 first, or --force for a deliberate exception")
    lock = acquire_lock(out, args.force)
    try:
        write_grid_manifest(out, args.grid, args.order, cells, budget, hyper)
        print(f"{len(cells)} cell(s), {args.concurrent} at a time, {args.threads} threads each, "
              f"out {out}, budget {budget} agent-steps, git {provenance['describe']}")
        start = time.perf_counter()
        results = launch(cells, out, args.concurrent, args.threads, args.agent_steps, extra)
        write_grid_manifest(out, args.grid, args.order, cells, budget, hyper)
        failed = [r for r in results if r["status"] != "complete"]
        print(f"finished {len(results)} run(s) in {(time.perf_counter() - start) / 60:.1f} min; {len(failed)} failed")
        if failed:
            for r in failed:
                print(f"  FAILED {r['condition']} seed {r['seed']} exit {r['exit_code']}")
            raise SystemExit(1)
    finally:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
