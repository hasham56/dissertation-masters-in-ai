"""Roll out any policy on fixed evaluation seeds and write one Parquet per episode.

The evaluation path steps :class:`hamlet.core.HamletCore` directly; it never
goes through PettingZoo. Every policy, hand-written or learned, is
driven by the same loop, so the logs are comparable row for row.

Seeds
-----
The evaluation seeds are ``EVAL_SEED_BASE + k`` for ``k`` in ``0 .. N_EVAL_SEEDS-1``
and are identical across every condition and policy. Episode 0 of seed ``s``
reseeds the core with ``s`` exactly; later episodes of the same seed use the
stream ``default_rng([s, episode])`` so that each file is reproducible on
its own. A policy that owns a ``rng`` attribute is reseeded the same way
before each episode.

Files
-----
``<out_dir>/<condition>/<policy_or_checkpoint>/seed<seed>_ep<k>.parquet``,
one row per (tick, agent), columns exactly ``config.LOG_COLUMNS``, burn-in
ticks included. ``t``, ``day``, ``t_day`` and ``action`` describe the
decision; every other field is the state after that tick. Beside the Parquet
files, ``traits_seed<seed>.csv`` records the traits drawn for that run seed
(one row per agent; see :mod:`hamlet.traits`).

Traits
------
The run seed is the trait seed: ``rollout`` builds the core with
``traits_seed=seed``, so every episode of a seed, and every policy evaluated
on it, sees the same agents. ``identity_perm`` replays an episode with traits
permuted across agents (the trait-and-home shuffle null of the
division-of-labour metric; homes are each agent's own zone and stay put) without
touching the world's random stream.

Command line
------------
``python -m hamlet.evaluate --policy GREEDY-CLOCK`` runs a baseline on the
32 evaluation seeds; ``--fallback FILE`` evaluates a checkpoint of the trainer.
"""
from __future__ import annotations

import argparse
import glob as globlib
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from hamlet.config import LOG_COLUMNS, N_ACTIONS, HamletConfig
from hamlet.core import HamletCore
from hamlet.policies import GreedyClockPolicy, GreedyClockWorkPolicy, GreedyStatePolicy, Policy, RandomPolicy
from hamlet.traits import POPULATIONS, Traits, sample_traits, write_traits_csv

# Evaluation seeds: the same 32 worlds for every condition and policy.
EVAL_SEED_BASE = 10_000
N_EVAL_SEEDS = 32

# Name of the checkpoint column for policies that have none.
NO_CHECKPOINT = "none"
# Decoding modes of a learned policy at evaluation: the primary and the secondary.
MODES = ("stochastic", "argmax")

# Builders of the hand-written baselines, keyed by policy name.
BASELINES: dict[str, Callable[[np.random.Generator], Policy]] = {
    "RANDOM": lambda rng: RandomPolicy(rng),
    "GREEDY-STATE": lambda rng: GreedyStatePolicy(),
    "GREEDY-CLOCK": lambda rng: GreedyClockPolicy(),
    "GREEDY-CLOCK-WORK": lambda rng: GreedyClockWorkPolicy(),
}

# Column dtypes of an evaluation log; everything else in LOG_COLUMNS is int64.
LOG_DTYPES: dict[str, Any] = {
    "condition": "str",
    "checkpoint": "str",
    "active": bool,
    "queued": bool,
    "informed": bool,
    # World v2: the two tanks are levels like the states; meal_type is a short string; and
    # company_others is a count, so it falls through to the int64 default below.
    "meal_type": "str",
    **{c: np.float32 for c in ("E", "F", "C", "fast_tank", "slow_tank",
                               "W", "M", "D", "reward", "effort_penalty", "rest_idle_cost",
                               "coins_spent_food", "coins_spent_drink")},
    "drank": bool,
    **{f"logp{k}": np.float32 for k in range(N_ACTIONS)},
}


def evaluation_seeds(n: int = N_EVAL_SEEDS) -> list[int]:
    """The first ``n`` evaluation seeds, ``EVAL_SEED_BASE + k``."""
    return [EVAL_SEED_BASE + k for k in range(n)]


def episode_rng(seed: int, episode: int) -> np.random.Generator:
    """Generator for episode ``episode`` of evaluation seed ``seed``."""
    return np.random.default_rng(seed if episode == 0 else [seed, episode])


def frame_from_columns(columns: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """Concatenate per-tick column arrays into a typed DataFrame in ``LOG_COLUMNS`` order."""
    data = {name: np.concatenate(columns[name]) for name in LOG_COLUMNS}
    df = pd.DataFrame(data, columns=LOG_COLUMNS)
    for name in LOG_COLUMNS:
        df[name] = df[name].astype(LOG_DTYPES.get(name, np.int64))
    return df


def rollout(
    cfg: HamletConfig,
    policy: Policy,
    seed: int,
    episode: int = 0,
    checkpoint: str = NO_CHECKPOINT,
    condition: Optional[str] = None,
    init_range: Optional[tuple[float, float]] = None,
    max_ticks: Optional[int] = None,
    identity_perm: Optional[np.ndarray] = None,
    traits_seed: Optional[int] = None,
    mode: str = "stochastic",
) -> pd.DataFrame:
    """Run one episode and return its log, ``episode_ticks * N`` rows in ``LOG_COLUMNS``.

    ``mode`` selects the decoding of a learned policy: ``"stochastic"`` (the
    primary evaluation, the policy's own sampled action) or ``"argmax"`` (the
    secondary evaluation, the argmax of the log-probability vector the policy
    returns; the world stream and the logged log-probabilities are unchanged).
    Scripted policies return no log-probabilities and are unaffected.

    ``condition`` defaults to ``cfg.condition_name``. ``init_range`` overrides
    the initial-state distribution (the response-threshold test). ``max_ticks``
    cuts the episode short, for smoke tests only. The run seed is passed to
    the core as ``traits_seed``; ``traits_seed`` overrides it (reports that
    hold one trait assignment fixed across evaluation seeds). ``identity_perm``
    (``int (N,)``) gives agent ``i`` the traits and the home of agent
    ``identity_perm[i]`` while every world draw stays the same.
    """
    condition = cfg.condition_name if condition is None else condition
    core = HamletCore(cfg, seed, traits_seed=seed if traits_seed is None else int(traits_seed))
    if identity_perm is not None:
        perm = np.asarray(identity_perm, dtype=np.int64)
        core.set_traits(core.traits.permuted(perm))
    core.rng = episode_rng(seed, episode)
    obs = core.reset(None, init_range=init_range)
    if hasattr(policy, "rng"):
        policy.rng = episode_rng(seed, episode)

    n_ticks = cfg.episode_ticks if max_ticks is None else min(int(max_ticks), cfg.episode_ticks)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in LOG_COLUMNS}
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    held_logps = None
    for _ in range(n_ticks):
        # The policy is consulted only where its action would be used, so training and evaluation
        # see the same decision points; on a journey tick the core holds the action and the stored
        # log-probabilities are those of the decision that started the journey.
        decision = core.can_decide
        if decision.any():
            actions, logps = policy.act(obs, core)
            if mode == "argmax" and logps is not None:
                actions = np.argmax(np.asarray(logps), axis=1).astype(np.int64)
            if logps is not None:
                held = np.asarray(logps, dtype=np.float32)
                held_logps = held if held_logps is None else np.where(decision[:, None], held, held_logps)
        else:
            actions = np.zeros(cfg.N, dtype=np.int64)      # ignored: every agent is mid-journey
        obs, rewards, _ = core.step(actions)
        cols = core.log_columns(seed, condition, checkpoint, episode, actions, rewards, held_logps)
        for name in LOG_COLUMNS:
            columns[name].append(cols[name])
    return frame_from_columns(columns)


def run_label(policy: Policy, checkpoint: str = NO_CHECKPOINT) -> str:
    """Directory name under the condition: the checkpoint label, else the policy name."""
    return policy.name if checkpoint == NO_CHECKPOINT else checkpoint


def episode_path(out_dir: Path, condition: str, label: str, seed: int, episode: int) -> Path:
    """``<out_dir>/<condition>/<label>/seed<seed>_ep<episode>.parquet``."""
    return Path(out_dir) / condition / label / f"seed{seed}_ep{episode}.parquet"


def write_log(df: pd.DataFrame, path: Path) -> Path:
    """Write an episode log as Parquet (pyarrow), creating the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def run_traits(cfg: HamletConfig, seed: int) -> list[Traits]:
    """The traits the core draws for run seed ``seed`` under ``cfg`` (one per agent)."""
    return sample_traits(cfg.population, cfg.N, seed, normalise=cfg.normalise_aptitude)


def traits_path(out_dir: Path, condition: str, label: str, seed: int) -> Path:
    """``<out_dir>/<condition>/<label>/traits_seed<seed>.csv``."""
    return Path(out_dir) / condition / label / f"traits_seed{seed}.csv"


def trait_and_home_shuffle_counts(
    cfg: HamletConfig,
    make_policy: Callable[[], Policy],
    seed: int,
    n_perm: int,
    rng: np.random.Generator,
    episode: int = 0,
    init_range: Optional[tuple[float, float]] = None,
) -> list[np.ndarray]:
    """Replay count matrices for the ``trait_and_home_shuffle`` null of the division of labour.

    Each of the ``n_perm`` replays rolls out a fresh policy from
    ``make_policy`` on the same seed and episode with one permutation drawn
    from ``rng`` applied to traits and homes (``rollout(identity_perm=...)``),
    and returns its ``(N, n_zones)`` time budget with rows in agent-id order,
    so that the part of the division of labour tied to traits and homes is
    moved between identities while the identities themselves stay put.
    """
    from hamlet.metrics.specialisation import time_budget

    counts = []
    for _ in range(int(n_perm)):
        perm = rng.permutation(cfg.N)
        df = rollout(cfg, make_policy(), seed, episode, init_range=init_range, identity_perm=perm)
        counts.append(time_budget(df, burn_in_days=cfg.burn_in_days, n_agents=cfg.N))
    return counts


def evaluate(
    cfg: HamletConfig,
    policy: Policy,
    seeds: Sequence[int],
    out_dir: Path,
    episodes_per_seed: int = 1,
    checkpoint: str = NO_CHECKPOINT,
    condition: Optional[str] = None,
    init_range: Optional[tuple[float, float]] = None,
    overwrite: bool = False,
    verbose: bool = False,
    mode: str = "stochastic",
) -> list[Path]:
    """Roll out ``policy`` on every seed and write one Parquet per (seed, episode).

    Existing files are skipped unless ``overwrite``. Returns the paths written
    or found, in seed-major order. ``traits_seed<seed>.csv`` is written beside
    the Parquet files of every seed. ``mode`` is passed to :func:`rollout`
    (``"stochastic"`` or ``"argmax"``).
    """
    condition = cfg.condition_name if condition is None else condition
    label = run_label(policy, checkpoint)
    paths: list[Path] = []
    for seed in seeds:
        csv_path = traits_path(out_dir, condition, label, seed)
        if overwrite or not csv_path.exists():
            write_traits_csv(csv_path, run_traits(cfg, seed), cfg.population)
        for k in range(episodes_per_seed):
            path = episode_path(out_dir, condition, label, seed, k)
            if path.exists() and not overwrite:
                paths.append(path)
                continue
            start = time.perf_counter()
            df = rollout(cfg, policy, seed, k, checkpoint, condition, init_range, mode=mode)
            write_log(df, path)
            paths.append(path)
            if verbose:
                print(f"{path}  ({time.perf_counter() - start:.1f} s)")
    return paths


def load_runs(pattern: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Concatenate the Parquet files matching ``pattern`` with a ``file`` column.

    ``pattern`` is a glob string (``**`` allowed), a directory (all of its
    Parquet files) or a sequence of either. Files are read in sorted order.
    """
    patterns = [pattern] if isinstance(pattern, (str, Path)) else list(pattern)
    files: list[str] = []
    for p in patterns:
        p = str(p)
        if Path(p).is_dir():
            p = str(Path(p) / "*.parquet")
        files.extend(globlib.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no Parquet files match {pattern!r}")
    frames = []
    for f in files:
        df = pd.read_parquet(f, engine="pyarrow")
        df["file"] = f
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_policy(
    name: Optional[str] = None,
    fallback: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, str]:
    """Build a policy from command-line options; returns ``(policy, checkpoint_label)``.

    Exactly one of ``name`` (a baseline) or ``fallback`` (a ``.pt`` checkpoint of the trainer) is used.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if fallback is not None:
        from hamlet.policies import FallbackPolicy

        path = Path(fallback)
        return FallbackPolicy(path, rng), path.stem
    if name is None or name not in BASELINES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(BASELINES)}")
    return BASELINES[name](rng), NO_CHECKPOINT


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Command-line options that select the evaluation condition."""
    parser.add_argument("--arm", default="A", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--symmetry", default="S1", choices=["S0", "S1"])
    parser.add_argument("--n-agents", type=int, default=HamletConfig.n_agents)
    parser.add_argument("--noclock", action="store_true", help="clock block zero-filled")
    parser.add_argument("--social-off", action="store_true", help="social need switched off")
    parser.add_argument("--mood", action="store_true", help="mood couples into productivity")
    parser.add_argument("--shock", default=None, choices=[None, "jobs_closed_d4", "energy_x2_d4"])
    parser.add_argument("--population", default="neutral", choices=sorted(POPULATIONS), help="trait variant")
    parser.add_argument("--social-solo-fraction", type=float, default=HamletConfig.social_solo_fraction,
                        help="exploratory (arm E2): fraction of social_gain an agent active alone at "
                             "SOCIAL receives; 0.0 is the registered world")


def config_from_args(args: argparse.Namespace) -> HamletConfig:
    """The :class:`HamletConfig` described by :func:`add_config_arguments`."""
    return HamletConfig(
        n_agents=args.n_agents,
        arm=args.arm,
        symmetry=args.symmetry,
        clock_visible=not args.noclock,
        social_on=not args.social_off,
        mood_coupling=args.mood,
        shock=args.shock,
        population=args.population,
        social_solo_fraction=getattr(args, "social_solo_fraction", HamletConfig.social_solo_fraction),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default=None, choices=sorted(BASELINES), help="a hand-written baseline")
    parser.add_argument("--fallback", default=None, help="fallback-trainer checkpoint (.pt)")
    parser.add_argument("--n-seeds", type=int, default=N_EVAL_SEEDS)
    parser.add_argument("--episodes", type=int, default=1, help="episodes per seed")
    parser.add_argument("--out", default="runs", help="root of the runs directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", default="stochastic", choices=MODES, help="decoding of a learned policy: stochastic (primary) or argmax (secondary)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    policy, checkpoint = make_policy(args.policy, args.fallback)
    if args.mode != "stochastic":
        checkpoint = f"{checkpoint}_{args.mode}"       # the secondary decoding never shares the primary's directory
    start = time.perf_counter()
    paths = evaluate(
        cfg, policy, evaluation_seeds(args.n_seeds), Path(args.out),
        episodes_per_seed=args.episodes, checkpoint=checkpoint, overwrite=args.overwrite, verbose=True, mode=args.mode,
    )
    print(f"{len(paths)} files under {paths[0].parent} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
