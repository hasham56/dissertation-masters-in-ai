"""Unit tests of the ``SandboxEnv`` levers against hand-computed values, and of the config validator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hamlet.config import HamletConfig

from sandbox.core import SandboxEnv, need_array
from sandbox.train import validate_config

ROOT = Path(__file__).resolve().parents[2]
# state arrays are sized by the configured population rather than a literal, so a population change cannot stale them
NN = HamletConfig().n_agents

NEUTRAL = json.loads((ROOT / "sandbox" / "configs" / "neutral_anchor.json").read_text())


def _set_states(env: SandboxEnv, E, F, C) -> None:
    env.E = np.asarray(E, dtype=np.float64)
    env.F = np.asarray(F, dtype=np.float64)
    env.C = np.asarray(C, dtype=np.float64)


def test_need_array_broadcasts_and_overrides():
    arr = need_array({"satiety": 0.8}, {"2": {"social": 0.6}, 5: {"energy": 0.5, "social": 0.4}}, 8, 1.0)
    assert arr.shape == (8, 3)
    assert arr[0].tolist() == [1.0, 0.8, 1.0]
    assert arr[2].tolist() == [1.0, 0.8, 0.6]
    assert arr[5].tolist() == [0.5, 0.8, 0.4]
    with pytest.raises(ValueError):
        need_array({"hunger": 1.0}, None, 8, 1.0)
    with pytest.raises(ValueError):
        need_array({}, {"8": {"energy": 1.0}}, 8, 1.0)


def test_drive_hand_computed_symmetric_and_deficit():
    cfg = HamletConfig()
    sym = SandboxEnv(cfg, 0, set_points={"energy": 0.7, "satiety": 1.0, "social": 0.9},
                     drive_weights_per_agent={"1": {"satiety": 2.0}, "2": {"energy": 0.0}})
    E = np.full(NN, 0.5); F = np.full(NN, 0.8); C = np.full(NN, 0.95)
    _set_states(sym, E, F, C)
    d = sym._drive()
    # agent 0: (0.7-0.5)^2 + (1-0.8)^2 + (0.9-0.95)^2 = 0.04 + 0.04 + 0.0025
    assert d[0] == pytest.approx(0.0825)
    # agent 1: satiety weight 2 -> 0.04 + 0.08 + 0.0025
    assert d[1] == pytest.approx(0.1225)
    # agent 2: energy weight 0 -> 0.04 + 0.0025
    assert d[2] == pytest.approx(0.0425)
    dfc = SandboxEnv(cfg, 0, set_points={"energy": 0.7, "satiety": 1.0, "social": 0.9}, set_point_form="deficit")
    _set_states(dfc, E, F, C)
    # social 0.95 sits above its set point 0.9: free under the deficit form
    assert dfc._drive()[0] == pytest.approx(0.08)


def test_overshoot_above_a_0_7_set_point_is_penalised_under_symmetric_and_free_under_deficit():
    cfg = HamletConfig()
    points = {"energy": 0.7, "satiety": 1.0, "social": 1.0}
    sym = SandboxEnv(cfg, 0, set_points=points, set_point_form="symmetric")
    dfc = SandboxEnv(cfg, 0, set_points=points, set_point_form="deficit")
    for env in (sym, dfc):
        _set_states(env, np.full(NN, 0.9), np.ones(NN), np.ones(NN))
    assert sym._drive()[0] == pytest.approx(0.04)
    assert dfc._drive()[0] == pytest.approx(0.0)
    for env in (sym, dfc):
        _set_states(env, np.full(NN, 0.5), np.ones(NN), np.ones(NN))
    assert sym._drive()[0] == pytest.approx(0.04) and dfc._drive()[0] == pytest.approx(0.04)


def test_neutral_levers_reduce_to_the_study_drive_bit_for_bit():
    cfg = HamletConfig()
    env = SandboxEnv(cfg, 3)
    rng = np.random.default_rng(1)
    _set_states(env, rng.random(NN), rng.random(NN), rng.random(NN))
    study = cfg.w_energy * (1.0 - env.E) ** 2 + cfg.w_satiety * (1.0 - env.F) ** 2 + cfg.w_social * (1.0 - env.C) ** 2
    assert np.array_equal(env._drive(), study)
    assert env.lever_summary()["neutral"] is True


def test_collective_lambda_mixes_the_reward_vector_and_preserves_its_sum():
    cfg = HamletConfig()
    env = SandboxEnv(cfg, 0, collective_lambda=0.5)
    obs = env.reset(0)
    plain = SandboxEnv(cfg, 0)
    plain.reset(0)
    for _ in range(50):
        actions = np.random.default_rng(7).integers(0, 7, size=NN)
        _, rew, _ = env.step(actions)
        _, raw_ref, _ = plain.step(actions)
        raw = env.last_raw_reward
        assert np.array_equal(raw, raw_ref)
        expected = (0.5 * raw + 0.5 * raw.mean()).astype(np.float32)
        assert np.array_equal(rew, expected)
        assert np.array_equal(env.last_shaped_reward, rew)
        assert float(rew.sum()) == pytest.approx(float(raw.sum()), abs=1e-5)
        # states and drive are untouched by the mixing
        assert np.array_equal(env.D, plain.D)
    summary = env.lever_summary()
    assert summary["collective_lambda"] == 0.5 and summary["neutral"] is False


def test_lever_validation_rejects_bad_values():
    cfg = HamletConfig()
    with pytest.raises(ValueError):
        SandboxEnv(cfg, 0, set_points={"energy": 0.0})
    with pytest.raises(ValueError):
        SandboxEnv(cfg, 0, set_points={"energy": 1.2})
    with pytest.raises(ValueError):
        SandboxEnv(cfg, 0, drive_weights_per_agent={"0": {"social": -1.0}})
    with pytest.raises(ValueError):
        SandboxEnv(cfg, 0, set_point_form="onesided")
    with pytest.raises(ValueError):
        SandboxEnv(cfg, 0, collective_lambda=1.5)


def test_config_validator_refuses_unknown_keys_and_missing_notes():
    ok = validate_config(dict(NEUTRAL))
    assert ok.is_neutral() and ok.name == "neutral_anchor"
    with pytest.raises(ValueError, match="unknown key"):
        validate_config({**NEUTRAL, "wage": 2.0})
    with pytest.raises(ValueError, match="notes"):
        validate_config({**NEUTRAL, "notes": "   "})
    with pytest.raises(ValueError, match="missing"):
        validate_config({k: v for k, v in NEUTRAL.items() if k != "seeds"})
    with pytest.raises(ValueError):
        validate_config({**NEUTRAL, "base": "A-S1-N8-nightowl"})
    with pytest.raises((ValueError, AssertionError)):
        validate_config({**NEUTRAL, "base": "Z-S1-N8"})
    with pytest.raises(ValueError):
        validate_config({**NEUTRAL, "set_points": {"energy": 1.0, "satiety": 1.0}})
    with pytest.raises(ValueError):
        validate_config({**NEUTRAL, "population": "giants"})
    with pytest.raises(ValueError):
        validate_config({**NEUTRAL, "hyper": {"learning_rate": 1e-3}})
