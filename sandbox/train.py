"""Train a ``SandboxEnv`` policy from an experiment config, reusing the fallback PPO by import.

Command line (the only entry point)::

    uv run python -m sandbox.train sandbox/configs/smoke.json
    uv run python -m sandbox.train sandbox/configs/exploring_social.json --seeds 0
    uv run python -m sandbox.train CONFIG [--seeds S [S ...]] [--out sandbox/runs] [--threads T]
                                          [--n-envs B] [--updates U] [--smoke] [--dry-run] [--no-eval]

Reuse by import, and the one thing that cannot be reused
--------------------------------------------------------
``hamlet.train_fallback.train`` builds ``BatchedCore`` by name and
``BatchedCore.__init__`` builds ``HamletCore`` by name; neither takes a core
class or factory, and hamlet's core is never patched from here. So this module owns two
things and imports everything else:

- ``SandboxBatchedCore(BatchedCore)``: overrides ``__init__`` only, to build
  ``SandboxEnv`` cores from the config's levers; ``reset``, ``step`` and
  ``pop_episode_returns`` are inherited.
- ``train``: the outer loop of ``hamlet.train_fallback.train`` written against
  ``SandboxBatchedCore``, with the same seeds (``env_seed * ENV_SEED_STRIDE + b``
  for the cores, ``policy_seed(env_seed)`` for torch), the same rollout length
  ``T = train_batch_size // (n_envs * N)`` and the same order of draws, so
  that a neutral config reproduces the study trainer checkpoint for
  checkpoint (``sandbox/tests/test_parity.py``). Every piece is imported:
  ``Actor``, ``Critic``, ``compute_gae``, ``ppo_update``, ``save_checkpoint``,
  ``ENV_SEED_STRIDE`` (train_fallback); ``PPOHyperparameters``,
  ``CheckpointRecord``, ``ProgressLog``, ``policy_seed``, ``write_metadata``,
  ``write_selection`` (train_common); ``iqm`` (metrics.stats).

Evaluation of the trained policy needs a rollout that steps ``SandboxEnv``;
``hamlet.evaluate.rollout`` builds ``HamletCore`` by name, so
``sandbox_rollout`` repeats that loop with ``SandboxEnv`` and reuses
``hamlet.evaluate.frame_from_columns`` and ``episode_rng`` so the Parquet has
exactly ``LOG_COLUMNS`` and every ``hamlet.metrics`` function reads it.

Logging honesty: the ``reward`` column of an
evaluation Parquet is always the raw homeostatic reward the core computed
(``-D`` minus the effort penalty, which is zero on the neutral population);
when ``collective_lambda > 0`` the shaped per-tick reward the policy was
trained on goes to ``shaped_reward.parquet`` beside the evaluation files.
``-D`` is therefore recoverable from every log and the shaped reward never
masquerades as it.

Outputs, in ``sandbox/runs/<config name>/seed<s>/``: ``metadata.json`` (the
study's fields plus ``exploratory``, the config's ``notes``, the resolved
``lever_summary`` and the git describe string), ``config.json`` (the config as
loaded), ``lever_summary.json``, ``traits.csv``, ``progress.csv`` (one row per
update: the training-return curve), ``ckpt_NNNN.pt``, ``selection.json``,
and after evaluation ``eval/seed<k>_ep0.parquet`` (plus
``shaped_reward.parquet`` when shaped).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from hamlet.config import LOG_COLUMNS, HamletConfig
from hamlet.evaluate import NO_CHECKPOINT, episode_rng, frame_from_columns, write_log
from hamlet.metrics.stats import iqm
from hamlet.policies import Policy
from hamlet.train_common import (
    CheckpointRecord,
    PPOHyperparameters,
    ProgressLog,
    policy_seed,
    write_metadata,
    write_selection,
)
from hamlet.train_fallback import (
    ENV_SEED_STRIDE,
    Actor,
    BatchedCore,
    Critic,
    compute_gae,
    ppo_update,
    save_checkpoint,
)
from hamlet.traits import POPULATIONS, TraitArrays, Traits, sample_traits, write_traits_csv

from sandbox import EXPLORATORY_TAG, RUNS_DIR, SANDBOX_ROOT, assert_sandbox_output, git_describe
from sandbox.core import NEEDS, SET_POINT_FORMS, SandboxEnv

TRAINER_NAME = "sandbox-fallback"
DEFAULT_EVAL_SEEDS = (10_000, 10_001, 10_002)
DEFAULT_N_ENVS = 4
DEFAULT_THREADS = 2
SMOKE_N_ENVS = 2
SMOKE_UPDATES = 1
SMOKE_EPOCHS = 2
CONFIGS_DIR = SANDBOX_ROOT / "configs"

MANDATORY_KEYS = ("name", "base", "set_points", "drive_weights", "population", "collective_lambda",
                  "seeds", "steps", "notes")
OPTIONAL_KEYS = ("set_point_form", "set_points_per_agent", "drive_weights_per_agent", "traits_seed",
                 "traits_override", "hyper", "eval_seeds")
WEIGHT_SUM_RANGE = (2.0, 4.0)


@dataclass(frozen=True)
class SandboxConfig:
    """One experiment config, validated by ``validate_config`` (via ``load_config``)."""

    name: str
    base: str
    set_points: dict[str, float]
    drive_weights: dict[str, float]
    population: str
    collective_lambda: float
    seeds: tuple[int, ...]
    steps: int
    notes: str
    set_point_form: str = "symmetric"
    set_points_per_agent: dict[str, dict[str, float]] = field(default_factory=dict)
    drive_weights_per_agent: dict[str, dict[str, float]] = field(default_factory=dict)
    traits_seed: Optional[int] = None
    traits_override: dict[str, dict[str, Any]] = field(default_factory=dict)
    hyper: dict[str, Any] = field(default_factory=dict)
    eval_seeds: tuple[int, ...] = DEFAULT_EVAL_SEEDS

    def hamlet_config(self) -> HamletConfig:
        """The ``HamletConfig`` for ``base`` with the population-wide drive weights and population applied."""
        cfg = parse_base(self.base)
        cfg = dataclasses.replace(
            cfg,
            w_energy=float(self.drive_weights["energy"]),
            w_satiety=float(self.drive_weights["satiety"]),
            w_social=float(self.drive_weights["social"]),
            population=self.population,
        )
        cfg.validate()
        return cfg

    def hyperparameters(self) -> PPOHyperparameters:
        """The study's ``PPOHyperparameters`` with ``total_agent_steps = steps`` and the ``hyper`` overrides."""
        hyper = PPOHyperparameters(total_agent_steps=int(self.steps))
        unknown = [k for k in self.hyper if k not in {f.name for f in dataclasses.fields(PPOHyperparameters)}]
        if unknown:
            raise ValueError(f"unknown hyperparameter override(s) {unknown}")
        overrides = {k: (tuple(v) if k == "hidden" else v) for k, v in self.hyper.items()}
        return dataclasses.replace(hyper, **overrides)

    def is_neutral(self) -> bool:
        return (
            all(float(v) == 1.0 for v in self.set_points.values())
            and all(float(v) == 1.0 for v in self.drive_weights.values())
            and self.population == "neutral"
            and float(self.collective_lambda) == 0.0
            and not self.set_points_per_agent
            and not self.drive_weights_per_agent
            and not self.traits_override
        )

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["seeds"] = list(self.seeds)
        out["eval_seeds"] = list(self.eval_seeds)
        return out


def parse_base(base: str) -> HamletConfig:
    """``"A-S1-N8"`` or ``"A-S1-N8-noclock"`` to a ``HamletConfig``; the result must round-trip to the same ``condition_name``."""
    parts = str(base).split("-")
    if len(parts) < 3 or not parts[2].startswith("N"):
        raise ValueError(f"base must look like 'A-S1-N8' or 'A-S1-N8-noclock'; got {base!r}")
    arm, symmetry, n = parts[0], parts[1], int(parts[2][1:])
    flags = parts[3:]
    kwargs: dict[str, Any] = {"arm": arm, "symmetry": symmetry, "n_agents": n}
    for flag in flags:
        if flag == "noclock":
            kwargs["clock_visible"] = False
        elif flag == "socialoff":
            kwargs["social_on"] = False
        else:
            raise ValueError(f"unknown base flag {flag!r} in {base!r}")
    cfg = HamletConfig(**kwargs)
    cfg.validate()
    if cfg.condition_name != base:
        raise ValueError(f"base {base!r} does not round-trip: resolved to {cfg.condition_name!r}")
    return cfg


def _check_need_map(name: str, mapping: Any, low: float, high: float, closed_low: bool) -> dict[str, float]:
    if not isinstance(mapping, dict) or set(mapping) != set(NEEDS):
        raise ValueError(f"{name} must map exactly {NEEDS}; got {mapping!r}")
    out = {}
    for need, value in mapping.items():
        v = float(value)
        ok = (v >= low if closed_low else v > low) and v <= high
        if not ok:
            raise ValueError(f"{name}[{need!r}] = {v} outside the valid range")
        out[need] = v
    return out


def _check_partial_map(name: str, mapping: Any, low: float, high: float, closed_low: bool) -> dict[str, dict[str, float]]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be an object of agent index -> partial need mapping")
    out: dict[str, dict[str, float]] = {}
    for key, partial in mapping.items():
        int(key)
        if not isinstance(partial, dict) or any(k not in NEEDS for k in partial):
            raise ValueError(f"{name}[{key!r}] must map a subset of {NEEDS}; got {partial!r}")
        row = {}
        for need, value in partial.items():
            v = float(value)
            ok = (v >= low if closed_low else v > low) and v <= high
            if not ok:
                raise ValueError(f"{name}[{key!r}][{need!r}] = {v} outside the valid range")
            row[need] = v
        out[str(key)] = row
    return out


def validate_config(raw: dict[str, Any]) -> SandboxConfig:
    """Validate a loaded JSON object (mandatory keys, ranges, no unknown keys) and build the config."""
    if not isinstance(raw, dict):
        raise ValueError("a config must be a JSON object")
    missing = [k for k in MANDATORY_KEYS if k not in raw]
    if missing:
        raise ValueError(f"config is missing mandatory key(s) {missing}")
    unknown = [k for k in raw if k not in MANDATORY_KEYS and k not in OPTIONAL_KEYS]
    if unknown:
        raise ValueError(f"config has unknown key(s) {unknown}; the accepted keys are MANDATORY_KEYS and OPTIONAL_KEYS")
    name = str(raw["name"])
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name):
        raise ValueError(f"name must match [A-Za-z0-9_-]+; got {name!r}")
    notes = raw["notes"]
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("notes must be a non-empty string stating what the setting is expected to do")
    set_points = _check_need_map("set_points", raw["set_points"], 0.0, 1.0, closed_low=False)
    drive_weights = _check_need_map("drive_weights", raw["drive_weights"], 0.0, float("inf"), closed_low=True)
    population = str(raw["population"])
    if population not in POPULATIONS:
        raise ValueError(f"population {population!r} is not one of {sorted(POPULATIONS)}")
    lam = float(raw["collective_lambda"])
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"collective_lambda must lie in [0, 1]; got {lam}")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds or any(int(s) != s for s in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    steps = raw["steps"]
    if int(steps) != steps or int(steps) <= 0:
        raise ValueError("steps must be a positive integer")
    form = str(raw.get("set_point_form", "symmetric"))
    if form not in SET_POINT_FORMS:
        raise ValueError(f"set_point_form must be one of {SET_POINT_FORMS}; got {form!r}")
    per_sp = _check_partial_map("set_points_per_agent", raw.get("set_points_per_agent", {}), 0.0, 1.0, False)
    per_w = _check_partial_map("drive_weights_per_agent", raw.get("drive_weights_per_agent", {}), 0.0, float("inf"), True)
    traits_seed = raw.get("traits_seed")
    if traits_seed is not None and int(traits_seed) != traits_seed:
        raise ValueError("traits_seed must be an integer or null")
    traits_override = raw.get("traits_override", {})
    if not isinstance(traits_override, dict):
        raise ValueError("traits_override must be an object of agent index -> partial trait mapping")
    for key, partial in traits_override.items():
        int(key)
        if not isinstance(partial, dict):
            raise ValueError(f"traits_override[{key!r}] must be an object")
        bad = [k for k in partial if k not in ("appetite", "metabolism", "chronotype", "laziness", "aptitude", "learning_rate")]
        if bad:
            raise ValueError(f"traits_override[{key!r}] has unknown trait(s) {bad}")
    hyper = raw.get("hyper", {})
    if not isinstance(hyper, dict):
        raise ValueError("hyper must be an object")
    eval_seeds = raw.get("eval_seeds", list(DEFAULT_EVAL_SEEDS))
    if not isinstance(eval_seeds, list) or not eval_seeds:
        raise ValueError("eval_seeds must be a non-empty list of integers")
    config = SandboxConfig(
        name=name, base=str(raw["base"]), set_points=set_points, drive_weights=drive_weights,
        population=population, collective_lambda=lam, seeds=tuple(int(s) for s in seeds), steps=int(steps),
        notes=notes, set_point_form=form, set_points_per_agent=per_sp, drive_weights_per_agent=per_w,
        traits_seed=None if traits_seed is None else int(traits_seed),
        traits_override={str(k): dict(v) for k, v in traits_override.items()}, hyper=dict(hyper),
        eval_seeds=tuple(int(s) for s in eval_seeds),
    )
    config.hamlet_config()      # base must parse and round-trip
    config.hyperparameters()    # hyper overrides must name real fields
    total = sum(drive_weights.values())
    if not WEIGHT_SUM_RANGE[0] <= total <= WEIGHT_SUM_RANGE[1]:
        print(f"warning: sum of drive weights {total:.2f} outside {WEIGHT_SUM_RANGE}: the reward scale moves with it")
    return config


def load_config(path: Path) -> SandboxConfig:
    """Read and validate one config JSON (mandatory keys, ranges, non-empty ``notes``, no unknown keys)."""
    return validate_config(json.loads(Path(path).read_text()))


def override_traits(cfg: HamletConfig, traits_seed: int, override: dict[str, dict[str, Any]]) -> TraitArrays:
    """The population's traits for ``traits_seed`` with per-agent fields replaced; caps enforced by ``validate``."""
    base = list(sample_traits(cfg.population, cfg.N, traits_seed, normalise=cfg.normalise_aptitude))
    for key, partial in override.items():
        agent = int(key)
        if not 0 <= agent < cfg.N:
            raise ValueError(f"traits_override agent {agent} outside range({cfg.N})")
        fields = dict(partial)
        if "aptitude" in fields:
            fields["aptitude"] = tuple(float(a) for a in fields["aptitude"])
        base[agent] = dataclasses.replace(base[agent], **fields).validate()
    return TraitArrays.from_traits(base)


def build_env(config: SandboxConfig, seed: int, traits_seed: Optional[int] = None) -> SandboxEnv:
    """One ``SandboxEnv`` for ``seed`` with every lever of ``config`` installed."""
    cfg = config.hamlet_config()
    ts = seed if traits_seed is None else int(traits_seed)
    traits = override_traits(cfg, ts, config.traits_override) if config.traits_override else None
    return SandboxEnv(
        cfg, int(seed), traits_seed=ts, traits=traits,
        set_points=config.set_points, set_points_per_agent=config.set_points_per_agent,
        set_point_form=config.set_point_form, drive_weights_per_agent=config.drive_weights_per_agent,
        collective_lambda=config.collective_lambda,
    )


class SandboxBatchedCore(BatchedCore):
    """``BatchedCore`` whose cores are ``SandboxEnv`` instances built from a ``SandboxConfig``."""

    def __init__(self, config: SandboxConfig, seeds: Sequence[int], traits_seed: Optional[int] = None) -> None:
        # the parent's attributes, built the parent's way, with SandboxEnv in place of HamletCore
        self.cfg = config.hamlet_config()
        self.cores = [build_env(config, int(s), traits_seed) for s in seeds]
        self.B = len(self.cores)
        self.N = self.cfg.N
        self.episode_returns = []
        self._running = np.zeros((self.B, self.N), dtype=np.float64)


def run_dir_for(config: SandboxConfig, env_seed: int, out_dir: Path = RUNS_DIR) -> Path:
    """``<out_dir>/<config.name>/seed<env_seed>``, checked to lie inside ``sandbox/runs/``."""
    return assert_sandbox_output(Path(out_dir) / config.name / f"seed{env_seed}")


def write_run_files(run_dir: Path, config: SandboxConfig, cfg: HamletConfig, hyper: PPOHyperparameters,
                    env_seed: int, envs: SandboxBatchedCore, extra: dict[str, Any]) -> None:
    """``metadata.json`` (via the study's writer), ``config.json``, ``lever_summary.json``, ``traits.csv``."""
    summary = envs.cores[0].lever_summary()
    write_metadata(run_dir, cfg, hyper, TRAINER_NAME, env_seed,
                   extra={**EXPLORATORY_TAG, "notes": config.notes, "sandbox_config": config.name,
                          "lever_summary": summary, "git": git_describe(), **extra})
    (run_dir / "config.json").write_text(json.dumps({**config.to_dict(), **EXPLORATORY_TAG,
                                                       "resolved_hamlet_config": json.loads(cfg.to_json())}, indent=2))
    (run_dir / "lever_summary.json").write_text(json.dumps({**summary, **EXPLORATORY_TAG}, indent=2))
    ts = int(envs.cores[0].traits_seed)
    if config.traits_override:
        arrays = envs.cores[0].traits
        rows = [Traits(float(arrays.appetite[i]), float(arrays.metabolism[i]), float(arrays.chronotype[i]),
                       float(arrays.laziness[i]), tuple(float(a) for a in arrays.aptitude[i]),
                       float(arrays.learning_rate[i])) for i in range(cfg.N)]
        write_traits_csv(run_dir / "traits.csv", rows, f"{cfg.population}+override")
    else:
        write_traits_csv(run_dir / "traits.csv",
                         sample_traits(cfg.population, cfg.N, ts, normalise=cfg.normalise_aptitude), cfg.population)


def train(
    config: SandboxConfig,
    env_seed: int,
    out_dir: Path = RUNS_DIR,
    n_envs: int = DEFAULT_N_ENVS,
    max_updates: Optional[int] = None,
    threads: int = DEFAULT_THREADS,
    verbose: bool = True,
    hyper: Optional[PPOHyperparameters] = None,
) -> Path:
    """Train one seed and return ``<out_dir>/<config.name>/seed<env_seed>/``.

    Mirrors ``hamlet.train_fallback.train`` step for step (rollout buffers of
    ``T = train_batch_size // (n_envs * N)`` ticks, ``compute_gae``,
    ``ppo_update``, one ``progress.csv`` row per update, a checkpoint every
    ``checkpoint_every_updates`` updates, ``write_selection`` at the end)
    with ``SandboxBatchedCore`` in place of ``BatchedCore``.
    """
    import torch
    from torch.distributions import Categorical

    cfg = config.hamlet_config()
    hyper = config.hyperparameters() if hyper is None else hyper
    torch.set_num_threads(threads)
    n_agents = cfg.N
    if hyper.train_batch_size % (n_envs * n_agents) != 0:
        raise ValueError("train_batch_size must be a multiple of n_envs * n_agents")
    T = hyper.train_batch_size // (n_envs * n_agents)
    M = n_envs * n_agents

    run_dir = run_dir_for(config, env_seed, out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    traits_seed = env_seed if config.traits_seed is None else int(config.traits_seed)
    log = ProgressLog(run_dir / "progress.csv")

    torch.manual_seed(policy_seed(env_seed))
    shuffle_rng = torch.Generator().manual_seed(policy_seed(env_seed))
    actor = Actor(cfg.obs_dim, hyper.hidden, hyper.activation)
    critic = Critic(cfg.obs_dim, hyper.hidden, hyper.activation)
    optimiser = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=hyper.lr, eps=hyper.adam_eps)

    envs = SandboxBatchedCore(config, [env_seed * ENV_SEED_STRIDE + b for b in range(n_envs)], traits_seed=traits_seed)
    write_run_files(run_dir, config, cfg, hyper, env_seed, envs,
                    extra={"n_envs": n_envs, "rollout_ticks": T, "threads": threads, "traits_seed": traits_seed})
    obs = envs.reset().reshape(M, cfg.obs_dim)

    buf_obs = np.zeros((T, M, cfg.obs_dim), dtype=np.float32)
    buf_act = np.zeros((T, M), dtype=np.int64)
    buf_logp = np.zeros((T, M), dtype=np.float32)
    buf_val = np.zeros((T, M), dtype=np.float32)
    buf_next_val = np.zeros((T, M), dtype=np.float32)
    buf_rew = np.zeros((T, M), dtype=np.float32)
    buf_done = np.zeros((T, M), dtype=np.float32)
    buf_dec = np.zeros((T, M), dtype=np.float32)   # mirrors the study trainer's decision mask

    records: list[CheckpointRecord] = []
    returns_since_ckpt: list[float] = []
    agent_steps = 0
    update = 0
    start = time.perf_counter()
    while agent_steps < hyper.total_agent_steps and (max_updates is None or update < max_updates):
        t_sample = time.perf_counter()
        drive_sum, drive_n = 0.0, 0     # reset each update: mean_D is this batch's mean
        with torch.no_grad():
            for t in range(T):
                obs_t = torch.from_numpy(obs)
                dist = Categorical(logits=actor(obs_t))
                actions = dist.sample()
                buf_dec[t] = envs.decision_mask().reshape(M)
                drive_sum += envs.mean_drive(); drive_n += 1
                buf_obs[t] = obs
                buf_act[t] = actions.numpy()
                buf_logp[t] = dist.log_prob(actions).numpy()
                buf_val[t] = critic(obs_t).numpy()
                next_obs, rew, trunc, final_obs = envs.step(buf_act[t].reshape(n_envs, n_agents))
                buf_rew[t] = rew.reshape(M)
                done = np.repeat(trunc, n_agents).astype(np.float32)
                buf_done[t] = done
                obs = next_obs.reshape(M, cfg.obs_dim)
                boot = np.where(done[:, None] > 0, final_obs.reshape(M, cfg.obs_dim), obs)
                buf_next_val[t] = critic(torch.from_numpy(np.ascontiguousarray(boot))).numpy()
        adv, ret = compute_gae(buf_rew, buf_val, buf_next_val, buf_done, hyper.gamma, hyper.gae_lambda)
        sample_time = time.perf_counter() - t_sample

        t_learn = time.perf_counter()
        batch = {
            "obs": torch.from_numpy(buf_obs.reshape(T * M, cfg.obs_dim)),
            "actions": torch.from_numpy(buf_act.reshape(T * M)),
            "logp": torch.from_numpy(buf_logp.reshape(T * M)),
            "adv": torch.from_numpy(adv.reshape(T * M).astype(np.float32)),
            "returns": torch.from_numpy(ret.reshape(T * M).astype(np.float32)),
            "decision": torch.from_numpy(buf_dec.reshape(T * M)),
            "vf_old": torch.from_numpy(buf_val.reshape(T * M).astype(np.float32)),
        }
        stats = ppo_update(actor, critic, optimiser, batch, hyper, shuffle_rng)
        learn_time = time.perf_counter() - t_learn

        update += 1
        agent_steps += T * M
        finished = envs.pop_episode_returns()
        returns_since_ckpt.extend(finished)
        row = {
            "update": update,
            "agent_steps": agent_steps,
            "episodes": len(finished),
            "episode_return_iqm": iqm(np.asarray(finished)) if finished else float("nan"),
            "episode_return_mean": float(np.mean(finished)) if finished else float("nan"),
            "mean_reward": float(buf_rew.mean()),
            "mean_D": drive_sum / max(drive_n, 1),   # batch mean, as in the study trainer
            **stats,
            "sample_s": sample_time,
            "learn_s": learn_time,
            "elapsed_s": time.perf_counter() - start,
        }
        log.write(row)
        if verbose:
            print(f"update {update:4d}  steps {agent_steps:>9d}  return {row['episode_return_mean']:9.2f}  "
                  f"entropy {stats['entropy']:.3f}  {sample_time:.1f}s+{learn_time:.1f}s")

        if update % hyper.checkpoint_every_updates == 0:
            path = save_checkpoint(run_dir / f"ckpt_{update:04d}.pt", actor, critic, cfg, hyper, update, agent_steps)
            records.append(CheckpointRecord(str(path), update, agent_steps, returns_since_ckpt))
            returns_since_ckpt = []

    if not records or records[-1].update != update:
        path = save_checkpoint(run_dir / f"ckpt_{update:04d}.pt", actor, critic, cfg, hyper, update, agent_steps)
        records.append(CheckpointRecord(str(path), update, agent_steps, returns_since_ckpt))
    write_selection(run_dir, records)
    (run_dir / "timing.json").write_text(json.dumps({
        "updates": update, "agent_steps": agent_steps, "wall_clock_s": time.perf_counter() - start,
        "agent_steps_per_s": agent_steps / max(time.perf_counter() - start, 1e-9), **EXPLORATORY_TAG}, indent=2))
    return run_dir


def sandbox_rollout(
    config: SandboxConfig,
    policy: Policy,
    seed: int,
    episode: int = 0,
    checkpoint: str = NO_CHECKPOINT,
    traits_seed: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One evaluation episode of ``policy`` on ``SandboxEnv``.

    Returns ``(log, shaped)``: ``log`` is a ``LOG_COLUMNS`` frame exactly as
    ``hamlet.evaluate.rollout`` builds it, its ``reward`` column the raw
    homeostatic reward; ``shaped`` has one row per (tick, agent) with
    ``seed, episode, t, agent, raw_reward, shaped_reward`` (the two columns
    are equal when ``collective_lambda == 0``).
    """
    cfg = config.hamlet_config()
    condition = cfg.condition_name
    ts = seed if traits_seed is None else int(traits_seed)
    if traits_seed is None and config.traits_seed is not None:
        ts = int(config.traits_seed)
    core = build_env(config, seed, ts)
    core.rng = episode_rng(seed, episode)
    obs = core.reset(None)
    if hasattr(policy, "rng"):
        policy.rng = episode_rng(seed, episode)

    columns: dict[str, list[np.ndarray]] = {name: [] for name in LOG_COLUMNS}
    shaped_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    ticks: list[int] = []
    held_logps = None
    for _ in range(cfg.episode_ticks):
        # mirrors hamlet.evaluate.rollout: consult the policy only where its action is used
        decision = core.can_decide
        if decision.any():
            actions, logps = policy.act(obs, core)
            if logps is not None:
                held = np.asarray(logps, dtype=np.float32)
                held_logps = held if held_logps is None else np.where(decision[:, None], held, held_logps)
        else:
            actions = np.zeros(cfg.N, dtype=np.int64)
        t_decision = core.t
        obs, rewards, _ = core.step(actions)
        raw = core.last_raw_reward
        cols = core.log_columns(seed, condition, checkpoint, episode, actions, raw, held_logps)
        for name in LOG_COLUMNS:
            columns[name].append(cols[name])
        raw_rows.append(np.asarray(raw, dtype=np.float32))
        shaped_rows.append(np.asarray(rewards, dtype=np.float32))
        ticks.append(t_decision)
    log = frame_from_columns(columns)
    n = cfg.N
    shaped = pd.DataFrame({
        "seed": np.int64(seed), "episode": np.int64(episode),
        "t": np.repeat(np.asarray(ticks, dtype=np.int64), n),
        "agent": np.tile(np.arange(n, dtype=np.int64), len(ticks)),
        "raw_reward": np.concatenate(raw_rows), "shaped_reward": np.concatenate(shaped_rows),
    })
    return log, shaped


def load_policy(run_dir: Path, rng: Optional[np.random.Generator] = None) -> tuple[Policy, str]:
    """The selected checkpoint of ``run_dir`` as a ``FallbackPolicy``; returns ``(policy, checkpoint label)``."""
    from hamlet.policies import FallbackPolicy

    selection = json.loads((Path(run_dir) / "selection.json").read_text())
    path = Path(selection["selected"])
    return FallbackPolicy(path, rng if rng is not None else np.random.default_rng(0)), path.stem


def evaluate_run(config: SandboxConfig, run_dir: Path, eval_seeds: Optional[Sequence[int]] = None,
                 episodes_per_seed: int = 1, verbose: bool = False) -> list[Path]:
    """Roll the selected checkpoint out on ``eval_seeds`` into ``run_dir/eval/``; returns the Parquet paths.

    When ``config.collective_lambda > 0`` the shaped rewards of every episode
    are concatenated into ``run_dir/shaped_reward.parquet``.
    """
    run_dir = assert_sandbox_output(Path(run_dir))
    seeds = list(config.eval_seeds if eval_seeds is None else eval_seeds)
    policy, label = load_policy(run_dir)
    paths: list[Path] = []
    shaped_frames: list[pd.DataFrame] = []
    for seed in seeds:
        for k in range(episodes_per_seed):
            start = time.perf_counter()
            log, shaped = sandbox_rollout(config, policy, seed, k, checkpoint=label)
            path = run_dir / "eval" / f"seed{seed}_ep{k}.parquet"
            write_log(log, path)
            paths.append(path)
            shaped_frames.append(shaped)
            if verbose:
                print(f"{path}  ({time.perf_counter() - start:.1f} s)")
    if config.collective_lambda > 0.0:
        pd.concat(shaped_frames, ignore_index=True).to_parquet(run_dir / "shaped_reward.parquet", engine="pyarrow", index=False)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment JSON under sandbox/configs/")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="subset of the config's seeds")
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="must lie inside sandbox/runs/")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS, help="cores stepped per update")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch CPU threads")
    parser.add_argument("--updates", type=int, default=None, help="stop after this many updates")
    parser.add_argument("--smoke", action="store_true", help="two cores, one small update")
    parser.add_argument("--dry-run", action="store_true", help="validate the config, print the resolved levers, train nothing")
    parser.add_argument("--no-eval", action="store_true", help="skip the evaluation rollouts after training")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out = assert_sandbox_output(args.out)
    seeds = list(config.seeds) if args.seeds is None else [int(s) for s in args.seeds]
    unknown = [s for s in seeds if s not in config.seeds]
    if unknown:
        raise ValueError(f"seed(s) {unknown} are not in the config's seeds {list(config.seeds)}")
    if args.dry_run:
        env = build_env(config, seeds[0], config.traits_seed)
        print(json.dumps({"config": config.to_dict(), "lever_summary": env.lever_summary(),
                          "condition": env.cfg.condition_name, "git": git_describe()}, indent=2, default=str))
        return
    hyper = config.hyperparameters()
    n_envs, max_updates = args.n_envs, args.updates
    if args.smoke:
        n_envs, max_updates = SMOKE_N_ENVS, SMOKE_UPDATES
        batch = config.hamlet_config().ticks_per_day * n_envs * config.hamlet_config().N
        hyper = dataclasses.replace(hyper, train_batch_size=batch, minibatch_size=batch // 2,
                                    num_epochs=SMOKE_EPOCHS, checkpoint_every_updates=1)
    for seed in seeds:
        start = time.perf_counter()
        run_dir = train(config, seed, out, n_envs=n_envs, max_updates=max_updates, threads=args.threads, hyper=hyper)
        print(f"finished {run_dir} in {time.perf_counter() - start:.1f} s")
        if not args.no_eval:
            paths = evaluate_run(config, run_dir, verbose=True)
            print(f"evaluated {len(paths)} episode(s) into {run_dir / 'eval'}")


if __name__ == "__main__":
    main()
