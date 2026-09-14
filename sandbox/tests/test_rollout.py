"""Rollout and logging honesty: the reward column is always the raw ``-D``; shaped reward goes to a sidecar."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hamlet.config import LOG_COLUMNS, HamletConfig
from hamlet.evaluate import LOG_DTYPES, rollout
from hamlet.policies import GreedyClockPolicy
from hamlet.train_common import PPOHyperparameters

from sandbox import RUNS_DIR
from sandbox.train import evaluate_run, sandbox_rollout, train, validate_config

ROOT = Path(__file__).resolve().parents[2]
NEUTRAL = json.loads((ROOT / "sandbox" / "configs" / "neutral_anchor.json").read_text())

# The trainer requires the batch to divide by n_envs * N, so the test batch is derived the same way
# the study's is: ticks_per_day * n_envs * N, one simulated day per environment. It moves with the
# population (2 envs x 9 agents x 240 ticks = 4,320; it was 3,840 at N = 8).
from hamlet.config import HamletConfig  # noqa: E402

_CFG = HamletConfig()
TEST_ENVS = 2
TEST_BATCH = _CFG.ticks_per_day * TEST_ENVS * _CFG.N
TEST_DIR = RUNS_DIR / "_test_rollout"


def _shaped_config(name: str, lam: float = 0.5, seeds=(0,), steps: int = 3840):
    return validate_config({**NEUTRAL, "name": name, "collective_lambda": lam, "seeds": list(seeds),
                            "steps": steps, "eval_seeds": [10000, 10001],
                            "notes": "test only: shaped reward must live in the sidecar, never in the log"})


def test_shaped_rollout_keeps_the_raw_reward_in_the_log_and_the_shaped_one_beside_it():
    config = _shaped_config("_test_shaped")
    cfg = HamletConfig()
    study_df = rollout(cfg, GreedyClockPolicy(), 10_000, 0)
    log, shaped = sandbox_rollout(config, GreedyClockPolicy(), 10_000, 0)
    assert list(log.columns) == LOG_COLUMNS
    assert {c: str(log[c].dtype) for c in LOG_COLUMNS} == {c: str(study_df[c].dtype) for c in LOG_COLUMNS}
    # the world is the study's (lambda touches nothing but the returned reward): the log is identical
    differing = [c for c in LOG_COLUMNS if not study_df[c].equals(log[c])]
    assert differing == [], differing
    # the raw column is -D exactly (neutral population: effort penalty zero)
    assert np.array_equal(log["reward"].to_numpy(), (-log["D"]).to_numpy())
    assert np.array_equal(log["reward"].to_numpy(), (-(log["D"] + log["effort_penalty"])).to_numpy())
    # the sidecar carries the shaped reward, which differs from the raw one
    assert list(shaped.columns) == ["seed", "episode", "t", "agent", "raw_reward", "shaped_reward"]
    assert len(shaped) == len(log)
    assert np.array_equal(shaped["raw_reward"].to_numpy(), log["reward"].to_numpy())
    raw = shaped["raw_reward"].to_numpy().reshape(-1, cfg.N)
    expected = (0.5 * raw + 0.5 * raw.mean(axis=1, keepdims=True)).astype(np.float32)
    assert np.array_equal(shaped["shaped_reward"].to_numpy().reshape(-1, cfg.N), expected)
    assert not np.array_equal(shaped["shaped_reward"].to_numpy(), shaped["raw_reward"].to_numpy())


def test_evaluate_run_writes_log_columns_parquets_and_the_sidecar():
    config = _shaped_config("_test_shaped_run")
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    hyper = PPOHyperparameters(total_agent_steps=TEST_BATCH, train_batch_size=TEST_BATCH,
                               minibatch_size=TEST_BATCH // 2, num_epochs=1, checkpoint_every_updates=1)
    run_dir = train(config, 0, TEST_DIR, n_envs=TEST_ENVS, max_updates=1, threads=1, verbose=False, hyper=hyper)
    assert run_dir == (TEST_DIR / "_test_shaped_run" / "seed0").resolve()
    for name in ("metadata.json", "config.json", "lever_summary.json", "traits.csv", "progress.csv", "selection.json", "timing.json"):
        assert (run_dir / name).exists(), name
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["exploratory"] is True and meta["notes"] == config.notes and "git" in meta
    assert meta["lever_summary"]["collective_lambda"] == 0.5
    paths = evaluate_run(config, run_dir, eval_seeds=[10_000, 10_001])
    assert [p.name for p in paths] == ["seed10000_ep0.parquet", "seed10001_ep0.parquet"]
    sidecar = run_dir / "shaped_reward.parquet"
    assert sidecar.exists()
    for p in paths:
        df = pd.read_parquet(p)
        assert list(df.columns) == LOG_COLUMNS
        assert all(str(df[c].dtype) == str(pd.Series(dtype=LOG_DTYPES.get(c, np.int64)).dtype) or c in ("condition", "checkpoint")
                   for c in LOG_COLUMNS)
        assert np.array_equal(df["reward"].to_numpy(), (-(df["D"] + df["effort_penalty"])).to_numpy())
        assert str(df["checkpoint"].iloc[0]).startswith("ckpt_")
    shaped = pd.read_parquet(sidecar)
    assert sorted(shaped["seed"].unique().tolist()) == [10_000, 10_001]
    assert len(shaped) == sum(len(pd.read_parquet(p)) for p in paths)
    assert not np.array_equal(shaped["shaped_reward"].to_numpy(), shaped["raw_reward"].to_numpy())


def test_neutral_evaluate_run_writes_no_sidecar():
    config = validate_config({**NEUTRAL, "name": "_test_neutral_run", "seeds": [0], "steps": TEST_BATCH,
                              "eval_seeds": [10000], "notes": "test only"})
    hyper = PPOHyperparameters(total_agent_steps=TEST_BATCH, train_batch_size=TEST_BATCH,
                               minibatch_size=TEST_BATCH // 2, num_epochs=1, checkpoint_every_updates=1)
    run_dir = train(config, 0, TEST_DIR, n_envs=TEST_ENVS, max_updates=1, threads=1, verbose=False, hyper=hyper)
    evaluate_run(config, run_dir, eval_seeds=[10_000])
    assert not (run_dir / "shaped_reward.parquet").exists()


def test_outputs_outside_sandbox_runs_are_refused(tmp_path):
    config = _shaped_config("_test_outside")
    with pytest.raises(ValueError, match="sandbox/runs"):
        train(config, 0, tmp_path, n_envs=TEST_ENVS, max_updates=1, threads=1, verbose=False)
    with pytest.raises(ValueError, match="sandbox/runs"):
        train(config, 0, ROOT / "runs", n_envs=TEST_ENVS, max_updates=1, threads=1, verbose=False)
