"""Tests for the evaluation path: rollout logs, Parquet files and run loading."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hamlet.config import LOG_COLUMNS, N_ACTIONS, HamletConfig
from hamlet.evaluate import (
    EVAL_SEED_BASE,
    N_EVAL_SEEDS,
    evaluate,
    evaluation_seeds,
    load_runs,
    make_policy,
    rollout,
    write_log,
)
from hamlet.policies import GreedyClockPolicy, GreedyClockWorkPolicy, GreedyStatePolicy, RandomPolicy
from hamlet.train_common import CheckpointRecord, select_checkpoint

SMALL = dict(n_agents=4, n_days=2, burn_in_days=1)


def test_evaluation_seeds_are_fixed():
    seeds = evaluation_seeds()
    assert seeds[0] == EVAL_SEED_BASE == 10_000 and len(seeds) == N_EVAL_SEEDS == 32
    assert seeds == list(range(10_000, 10_032))


def test_rollout_shape_columns_and_dtypes():
    cfg = HamletConfig(**SMALL)
    df = rollout(cfg, GreedyClockPolicy(), seed=10_000, episode=0)
    assert list(df.columns) == LOG_COLUMNS
    assert len(df) == cfg.episode_ticks * cfg.N
    assert df["t"].min() == 0 and df["t"].max() == cfg.episode_ticks - 1
    assert set(df["agent"]) == set(range(cfg.N))
    assert (df.groupby("t").size() == cfg.N).all()
    assert df["condition"].iloc[0] == cfg.condition_name and df["checkpoint"].iloc[0] == "none"
    for col in ("seed", "episode", "t", "day", "t_day", "agent", "x", "y", "zone", "action"):
        assert df[col].dtype == np.int64, col
    for col in ("active", "queued", "informed"):
        assert df[col].dtype == bool, col
    for col in ("E", "F", "C", "W", "M", "D", "reward"):
        assert df[col].dtype == np.float32, col
    assert df[[f"logp{k}" for k in range(N_ACTIONS)]].isna().all().all()   # greedy has no distribution


def test_rollout_logs_probabilities_for_random_policy():
    cfg = HamletConfig(**SMALL)
    df = rollout(cfg, RandomPolicy(np.random.default_rng(3)), seed=10_001, episode=0)
    lp = df[[f"logp{k}" for k in range(N_ACTIONS)]].to_numpy(dtype=np.float64)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)
    assert df["day"].max() == cfg.n_days - 1


def test_rollout_is_reproducible_per_seed_and_episode():
    cfg = HamletConfig(**SMALL)
    a = rollout(cfg, RandomPolicy(np.random.default_rng(0)), 10_000, 0)
    b = rollout(cfg, RandomPolicy(np.random.default_rng(99)), 10_000, 0)   # policy rng is reseeded
    c = rollout(cfg, RandomPolicy(np.random.default_rng(0)), 10_000, 1)
    assert a.equals(b)
    assert not a.drop(columns=["episode"]).equals(c.drop(columns=["episode"]))


def test_parquet_round_trip(tmp_path):
    cfg = HamletConfig(**SMALL)
    df = rollout(cfg, GreedyStatePolicy(), 10_000, 0)
    path = write_log(df, tmp_path / "x.parquet")
    back = pd.read_parquet(path)
    assert back.equals(df)
    assert back.dtypes.equals(df.dtypes)


def test_evaluate_writes_expected_file_names_and_load_runs_concatenates(tmp_path):
    cfg = HamletConfig(**SMALL)
    seeds = evaluation_seeds(2)
    paths = evaluate(cfg, GreedyClockPolicy(), seeds, tmp_path, episodes_per_seed=2)
    expected = [tmp_path / cfg.condition_name / "GREEDY-CLOCK" / f"seed{s}_ep{k}.parquet" for s in seeds for k in range(2)]
    assert paths == expected and all(p.exists() for p in expected)
    again = evaluate(cfg, GreedyClockPolicy(), seeds, tmp_path, episodes_per_seed=2)   # skips existing files
    assert again == expected

    df = load_runs(tmp_path / cfg.condition_name / "GREEDY-CLOCK")
    assert "file" in df.columns and df["file"].nunique() == 4
    assert len(df) == 4 * cfg.episode_ticks * cfg.N
    assert set(df["seed"]) == set(seeds) and set(df["episode"]) == {0, 1}
    by_glob = load_runs(str(tmp_path / "**" / "seed10000_ep0.parquet"))
    assert len(by_glob) == cfg.episode_ticks * cfg.N
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "nothing")


def test_make_policy_names():
    for name in ("RANDOM", "GREEDY-STATE", "GREEDY-CLOCK", "GREEDY-CLOCK-WORK"):
        policy, checkpoint = make_policy(name)
        assert policy.name == name and checkpoint == "none"
    assert isinstance(make_policy("GREEDY-CLOCK-WORK")[0], GreedyClockWorkPolicy)
    with pytest.raises(ValueError):
        make_policy("UNKNOWN")


def test_checkpoint_selection_uses_iqm_of_last_five_with_ties_to_latest():
    recs = [CheckpointRecord(f"c{i}", i, 1000 * i, [float(v)] * 8) for i, v in enumerate([5, 1, 9, 3, 3, 3, 3])]
    out = select_checkpoint(recs)
    assert [c["path"] for c in out["candidates"]] == ["c2", "c3", "c4", "c5", "c6"]
    assert out["selected"] == "c2"
    tie = [CheckpointRecord(f"c{i}", i, 0, [1.0, 2.0, 3.0]) for i in range(3)]
    assert select_checkpoint(tie)["selected"] == "c2"
    empty = [CheckpointRecord("c0", 0, 0, []), CheckpointRecord("c1", 1, 0, [])]
    assert select_checkpoint(empty)["selected"] == "c1"
    json.dumps(out, default=str)
