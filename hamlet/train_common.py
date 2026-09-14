"""The trainer's hyperparameters, run metadata, progress log and checkpoint selection.

The world constants live in :mod:`hamlet.config`; the optimiser constants live here, beside
the writers of ``metadata.json``, ``progress.csv`` and ``selection.json`` that
:mod:`hamlet.train_fallback` uses for every run directory.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from hamlet.config import HamletConfig
from hamlet.metrics.stats import iqm
from hamlet.traits import sample_traits, write_traits_csv

# Training seeds: policy seed = POLICY_SEED_BASE + s for environment seed s.
POLICY_SEED_BASE = 100
# Number of trailing checkpoints among which the final one is selected.
SELECTION_WINDOW = 5
# Packages whose versions are pinned into every run's metadata.
PINNED_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow", "gymnasium", "pettingzoo", "torch")


@dataclass
class PPOHyperparameters:
    """Clipped-objective PPO settings.

    Sizes are in agent-steps: ``train_batch_size`` samples per update,
    split into ``minibatch_size`` chunks for ``num_epochs`` passes. The batch is
    ``ticks_per_day * n_envs * N``, one simulated day per environment per update, so it moves with
    the population: 240 x 16 x 8 = 30,720 at N = 8, and 240 x 16 x 9 = 34,560 at N = 9. It has
    to: the trainer requires the batch to be a multiple of ``n_envs * N``, and 30,720
    carries a single factor of three while 9 needs two, so at N = 9 no choice of ``n_envs`` divides
    it and training refuses to start. The quantity registered is the day per environment, not the
    literal 30,720.
    """

    lr: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 0.997          # raised from 0.995 after the re-pilot; the pre-listed Gate 2 revision, now spent
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    vf_clip_param: float = 10.0
    entropy_coeff: float = 0.03    # raised from 0.01: re-pilot seed 1 fell to 0.0003 nats by update 192
    vf_loss_coeff: float = 0.5
    grad_clip: float = 0.5
    train_batch_size: int = 34_560   # 240 ticks x 16 envs x 9 agents; 30,720 at N = 8
    minibatch_size: int = 3_840      # unchanged: it still divides the batch, now into 9 chunks
    num_epochs: int = 8
    hidden: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    total_agent_steps: int = 30_000_000   # raised from 10M: re-pilot seed 0 was still improving at 10M
    checkpoint_every_updates: int = 16      # 16 x 34,560 = 552,960 agent-steps at N = 9


def policy_seed(env_seed: int) -> int:
    """Policy (network initialisation) seed paired with an environment seed."""
    return POLICY_SEED_BASE + int(env_seed)


def git_provenance(repo_root: Optional[Path] = None) -> dict[str, str]:
    """``{"describe", "commit"}`` of the repository (``"unknown"`` outside one or without git)."""
    root = Path(__file__).resolve().parents[1] if repo_root is None else Path(repo_root)
    out = {"describe": "unknown", "commit": "unknown"}
    for key, cmd in (("describe", ["git", "describe", "--always", "--dirty", "--tags"]),
                     ("commit", ["git", "rev-parse", "HEAD"])):
        try:
            res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                out[key] = res.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def utc_now() -> str:
    """ISO-8601 UTC timestamp with a ``Z`` suffix."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def package_versions() -> dict[str, str]:
    """Installed versions of the pinned packages plus the interpreter."""
    out = {"python": platform.python_version()}
    for name in PINNED_PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run_directory(out_dir: Path, cfg: HamletConfig, env_seed: int, suffix: str = "") -> Path:
    """``<out_dir>/<condition_name><suffix>/seed<env_seed>``."""
    return Path(out_dir) / f"{cfg.condition_name}{suffix}" / f"seed{env_seed}"


def write_metadata(
    run_dir: Path,
    cfg: HamletConfig,
    hyper: PPOHyperparameters,
    trainer: str,
    env_seed: int,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write ``metadata.json``: world config, hyperparameters, seeds, versions, command line."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "trainer": trainer,
        "condition": cfg.condition_name,
        "env_seed": int(env_seed),
        "policy_seed": policy_seed(env_seed),
        "hamlet_config": json.loads(cfg.to_json()),
        "hyperparameters": asdict(hyper),
        "versions": package_versions(),
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_utc": utc_now(),
        "git": git_provenance(),
        "step_budget": int(hyper.total_agent_steps),
        "status": "running",
    }
    if extra:
        meta.update(extra)
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path


def finish_metadata(run_dir: Path, extra: Optional[dict[str, Any]] = None) -> Path:
    """Merge the end of a run into ``metadata.json``: ``finished_utc``, ``status`` and ``extra`` (wall-clock, steps)."""
    path = Path(run_dir) / "metadata.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update({"finished_utc": utc_now(), "status": "complete"})
    if extra:
        meta.update(extra)
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path


def write_traits(run_dir: Path, cfg: HamletConfig, traits_seed: int) -> Path:
    """Write ``traits.csv`` (one row per agent) for the run's trait seed and return its path.

    Every core of a training run, whatever its runner or batch offset, draws
    the same traits from ``traits_seed``; this file records them once.
    """
    traits = sample_traits(cfg.population, cfg.N, traits_seed, normalise=cfg.normalise_aptitude)
    return write_traits_csv(Path(run_dir) / "traits.csv", traits, cfg.population)


class ProgressLog:
    """One row per update in ``progress.csv``; the header is fixed by the first row.

    The first write of a run truncates any file left by an earlier run of the
    same directory, so a restarted run never carries two headers.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fields: Optional[list[str]] = None

    def write(self, row: dict[str, Any]) -> None:
        new = self._fields is None
        if new:
            self._fields = list(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w" if new else "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields)
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fields})


@dataclass
class CheckpointRecord:
    """A saved checkpoint and the per-episode training returns gathered since the previous one."""

    path: str
    update: int
    agent_steps: int
    episode_returns: list[float] = field(default_factory=list)

    @property
    def iqm_return(self) -> float:
        return iqm(np.asarray(self.episode_returns)) if self.episode_returns else float("nan")


def select_checkpoint(records: Sequence[CheckpointRecord], window: int = SELECTION_WINDOW) -> dict[str, Any]:
    """Pick, among the last ``window`` checkpoints, the highest IQM of training return.

    Ties (and all-NaN windows) go to the latest checkpoint. The rule uses the
    training return only, never a behavioural metric.
    """
    recent = list(records)[-window:]
    if not recent:
        return {"selected": None, "candidates": []}
    best = recent[-1]
    for rec in recent:
        if np.isfinite(rec.iqm_return) and (not np.isfinite(best.iqm_return) or rec.iqm_return > best.iqm_return):
            best = rec
    return {
        "selected": best.path,
        "rule": f"highest IQM of per-episode training return among the last {window} checkpoints; ties to the latest",
        "candidates": [
            {"path": r.path, "update": r.update, "agent_steps": r.agent_steps,
             "n_episodes": len(r.episode_returns), "iqm_return": r.iqm_return}
            for r in recent
        ],
    }


def write_selection(run_dir: Path, records: Sequence[CheckpointRecord]) -> Path:
    """Write ``selection.json`` for a finished run and return its path."""
    path = Path(run_dir) / "selection.json"
    path.write_text(json.dumps(select_checkpoint(records), indent=2, default=str))
    return path
