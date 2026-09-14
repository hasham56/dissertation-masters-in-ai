"""Neutral parity: the sandbox with every lever neutral is the study, bit for bit.

These three tests are the standing definition of "the sandbox is the study
plus levers". Each compares a neutral ``SandboxEnv`` path against the study's
own code on the same seeds:

1. trajectories: the pinned hashes of ``tests/fixtures/world_trajectories.json``
   (imported from the study's fixture, never copied) are reproduced by
   ``SandboxEnv`` under every baseline policy;
2. rollouts: ``sandbox_rollout`` is column-identical (values and dtypes) to
   ``hamlet.evaluate.rollout``;
3. training: ``sandbox.train.train`` with a neutral config reproduces
   ``hamlet.train_fallback.train`` checkpoint for checkpoint on a short run.

Outputs of test 3 go under ``sandbox/runs/_parity/``, never under ``runs/``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hamlet.config import LOG_COLUMNS, HamletConfig
from hamlet.evaluate import rollout
from hamlet.policies import GreedyClockPolicy, GreedyClockWorkPolicy, GreedyStatePolicy, RandomPolicy
from hamlet.train_common import PPOHyperparameters

from sandbox import RUNS_DIR
from sandbox.core import SandboxEnv
from sandbox.train import load_config, sandbox_rollout, train

ROOT = Path(__file__).resolve().parents[2]
NEUTRAL = ROOT / "sandbox" / "configs" / "neutral_anchor.json"
PARITY_DIR = RUNS_DIR / "_parity"


def _study_regression_module():
    """The study's regression test module, loaded from its file (tests/ is not a package)."""
    path = ROOT / "tests" / "test_regression_world.py"
    spec = importlib.util.spec_from_file_location("study_regression_world", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sandbox_trajectory_hash(policy, seed: int, days: int) -> str:
    """The fixture's hashing procedure with ``SandboxEnv`` (neutral) in place of ``HamletCore``."""
    cfg = HamletConfig()
    core = SandboxEnv(cfg, seed)
    obs = core.reset(seed)
    h = hashlib.sha256()
    for _ in range(days * cfg.ticks_per_day):
        a, _ = policy.act(obs, core)
        obs, rew, _ = core.step(a)
        h.update(core.pos.astype(np.int64).tobytes())
        h.update(np.round(np.stack([core.E, core.F, core.C, core.W, core.M]), 9).tobytes())
        h.update(a.astype(np.int64).tobytes())
        h.update(np.round(rew, 9).astype(np.float64).tobytes())
    return h.hexdigest()


def test_neutral_sandbox_env_reproduces_the_pinned_trajectory_hashes():
    study = _study_regression_module()
    expected = json.loads(study.FIXTURE.read_text())["hashes"]
    got = {f"{name}/seed{seed}": _sandbox_trajectory_hash(make(), seed, study.DAYS)
           for name, make in study.policies().items() for seed in study.SEEDS}
    assert set(got) == set(expected)
    mismatches = {k: (expected[k], got[k]) for k in got if got[k] != expected[k]}
    assert mismatches == {}, mismatches


@pytest.mark.parametrize("make_policy", [
    lambda: GreedyClockPolicy(), lambda: GreedyStatePolicy(), lambda: GreedyClockWorkPolicy(),
    lambda: RandomPolicy(np.random.default_rng(5)),
])
def test_neutral_sandbox_rollout_is_column_identical_to_the_study_rollout(make_policy):
    config = load_config(NEUTRAL)
    assert config.is_neutral()
    cfg = config.hamlet_config()
    assert cfg == HamletConfig()
    study_df = rollout(cfg, make_policy(), 10_000, 0)
    sandbox_df, shaped = sandbox_rollout(config, make_policy(), 10_000, 0)
    assert list(sandbox_df.columns) == LOG_COLUMNS == list(study_df.columns)
    dtype_drift = {c: (str(study_df[c].dtype), str(sandbox_df[c].dtype)) for c in LOG_COLUMNS
                   if study_df[c].dtype != sandbox_df[c].dtype}
    assert dtype_drift == {}, dtype_drift
    differing = [c for c in LOG_COLUMNS if not study_df[c].equals(sandbox_df[c])]
    assert differing == [], differing
    assert np.array_equal(shaped["raw_reward"].to_numpy(), shaped["shaped_reward"].to_numpy())


def _checkpoint_digest(path: Path) -> str:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for part in ("actor", "critic"):
        for key in sorted(ckpt[part]):
            h.update(key.encode())
            h.update(ckpt[part][key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_neutral_sandbox_training_reproduces_the_study_trainer_checkpoints():
    import torch
    from hamlet.train_fallback import train as study_train

    config = load_config(NEUTRAL)
    cfg = config.hamlet_config()
    n_envs, updates, threads = 2, 3, 2
    hyper = PPOHyperparameters(total_agent_steps=100_000, checkpoint_every_updates=1)
    study_out, sandbox_out = PARITY_DIR / "study", PARITY_DIR / "sandbox"
    for d in (study_out, sandbox_out):
        shutil.rmtree(d, ignore_errors=True)
    study_dir = study_train(cfg, hyper, 0, study_out, n_envs=n_envs, max_updates=updates, threads=threads, verbose=False)
    sandbox_dir = train(config, 0, sandbox_out, n_envs=n_envs, max_updates=updates, threads=threads, verbose=False, hyper=hyper)

    study_ckpts = sorted(study_dir.glob("ckpt_*.pt"))
    sandbox_ckpts = sorted(sandbox_dir.glob("ckpt_*.pt"))
    assert [p.name for p in study_ckpts] == [p.name for p in sandbox_ckpts] and len(study_ckpts) == updates
    digests = {}
    for a, b in zip(study_ckpts, sandbox_ckpts):
        ca = torch.load(a, map_location="cpu", weights_only=False)
        cb = torch.load(b, map_location="cpu", weights_only=False)
        for part in ("actor", "critic"):
            assert ca[part].keys() == cb[part].keys()
            unequal = [k for k in ca[part] if not torch.equal(ca[part][k], cb[part][k])]
            assert unequal == [], (a.name, part, unequal)
        assert ca["agent_steps"] == cb["agent_steps"] and ca["update"] == cb["update"]
        digests[a.name] = (_checkpoint_digest(a), _checkpoint_digest(b))
        assert digests[a.name][0] == digests[a.name][1]
    # the progress rows agree on everything but wall-clock
    import pandas as pd

    ps = pd.read_csv(study_dir / "progress.csv")
    pb = pd.read_csv(sandbox_dir / "progress.csv")
    timing = {"sample_s", "learn_s", "elapsed_s"}
    for col in ps.columns:
        if col in timing:
            continue
        assert np.allclose(ps[col].to_numpy(dtype=float), pb[col].to_numpy(dtype=float), equal_nan=True, rtol=0, atol=0), col
    (PARITY_DIR / "checkpoint_digests.json").write_text(json.dumps(
        {name: {"study": s, "sandbox": b} for name, (s, b) in digests.items()}, indent=2))
