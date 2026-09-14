"""The evaluation ``mode``: stochastic reproduces today's logs byte for byte; argmax is greedy decoding.

Skipped (with a message) while ``hamlet.evaluate.rollout`` has no ``mode`` parameter, so
the suite stays green if the edit has not been applied yet (it may not be applied while a
grid trains).
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hamlet.config import HamletConfig
from hamlet.evaluate import rollout, write_log
from hamlet.policies import GreedyClockPolicy, RandomPolicy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HAS_MODE = "mode" in inspect.signature(rollout).parameters
pytestmark = pytest.mark.skipif(not HAS_MODE, reason="hamlet.evaluate.rollout has no 'mode' parameter yet (edit deferred while the grid trains)")


def _stub_checkpoint(tmp_path: Path) -> Path:
    """A tiny untrained fallback checkpoint, so a learned policy exists without training."""
    import torch
    from hamlet.train_common import PPOHyperparameters
    from hamlet.train_fallback import Actor, Critic, save_checkpoint

    cfg = HamletConfig()
    torch.manual_seed(0)
    actor, critic = Actor(cfg.obs_dim), Critic(cfg.obs_dim)
    return save_checkpoint(tmp_path / "ckpt_0001.pt", actor, critic, cfg, PPOHyperparameters(), 1, 30_720)


def _digest(df: pd.DataFrame, path: Path) -> str:
    write_log(df, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stochastic_mode_reproduces_the_existing_rollout_byte_for_byte(tmp_path):
    cfg = HamletConfig()
    for make in (lambda: GreedyClockPolicy(), lambda: RandomPolicy(np.random.default_rng(3))):
        plain = rollout(cfg, make(), 10_000, 0, max_ticks=300)
        moded = rollout(cfg, make(), 10_000, 0, max_ticks=300, mode="stochastic")
        assert plain.equals(moded)
        assert _digest(plain, tmp_path / "a.parquet") == _digest(moded, tmp_path / "b.parquet")
    from hamlet.policies import FallbackPolicy
    ckpt = _stub_checkpoint(tmp_path)
    plain = rollout(cfg, FallbackPolicy(ckpt), 10_000, 0, max_ticks=300)
    moded = rollout(cfg, FallbackPolicy(ckpt), 10_000, 0, max_ticks=300, mode="stochastic")
    assert plain.equals(moded)
    assert _digest(plain, tmp_path / "c.parquet") == _digest(moded, tmp_path / "d.parquet")
    # The stored baseline Parquets predate the fixed-cost travel rule, so they are a
    # different world and are deliberately not compared. What must hold is that "stochastic" is a
    # no-op relative to calling rollout without a mode, which the assertions above check.


def test_argmax_mode_is_greedy_decoding_and_matches_the_script_wrapper(tmp_path):
    from hamlet.policies import FallbackPolicy
    from evaluate_grid import ArgmaxPolicy

    cfg = HamletConfig()
    ckpt = _stub_checkpoint(tmp_path)
    greedy = rollout(cfg, FallbackPolicy(ckpt), 10_000, 0, max_ticks=300, mode="argmax")
    logp = greedy[[f"logp{k}" for k in range(7)]].to_numpy()
    assert np.array_equal(greedy["action"].to_numpy(), np.argmax(logp, axis=1))
    wrapped = rollout(cfg, ArgmaxPolicy(FallbackPolicy(ckpt)), 10_000, 0, max_ticks=300)
    assert greedy.equals(wrapped)
    sampled = rollout(cfg, FallbackPolicy(ckpt), 10_000, 0, max_ticks=300)
    assert not greedy["action"].equals(sampled["action"])
    with pytest.raises(ValueError):
        rollout(cfg, FallbackPolicy(ckpt), 10_000, 0, max_ticks=10, mode="beam")
