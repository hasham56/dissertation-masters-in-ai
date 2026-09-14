"""The grid launcher's plan, tiers, filters, manifests and resume rule, without training anything."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_grid as rg  # noqa: E402

# The condition name carries the population size, so the tests follow it rather than
# pinning a literal: N is read from the registered config.
from hamlet.config import HamletConfig  # noqa: E402

NN = HamletConfig().n_agents


def test_grid_sizes_and_condition_names():
    sizes = {g: len(rg.plan(g)) for g in rg.GRIDS}
    assert sizes == {"confirmatory": 48, "minimum": 32, "minimum-noclock-dropped": 24, "full": 60}
    names = [c.name for c, _ in rg.plan("confirmatory", "interleaved")[:6]]
    assert names == [f"A-S1-N{NN}", f"D-S1-N{NN}", f"A-S0-N{NN}", f"A-S1-N{NN}-noclock", f"B-S1-N{NN}", f"C-S1-N{NN}"]
    assert [c.name for c, _ in rg.plan("confirmatory")[:4]] == names[:4]     # default order: the four minimum conditions first
    assert [c.name for c, _ in rg.plan("minimum-noclock-dropped")[:3]] == [f"A-S1-N{NN}", f"D-S1-N{NN}", f"A-S0-N{NN}"]


def test_default_order_is_the_plan_order_and_interleaved_is_seed_major_over_all_six():
    assert rg.DEFAULT_ORDER == "minimum-first" and rg.plan("confirmatory") == rg.plan("confirmatory", "minimum-first")
    cells = rg.plan("confirmatory", "interleaved")
    assert [s for _, s in cells[:6]] == [0] * 6 and [s for _, s in cells[6:12]] == [1] * 6
    assert cells[-1][1] == 7 and cells[-1][0].name == f"C-S1-N{NN}"


def test_minimum_first_order_matches_the_plan_section_3():
    cells = rg.plan("confirmatory", "minimum-first")
    first32 = [c.name for c, _ in cells[:32]]
    assert set(first32) == {f"A-S1-N{NN}", f"D-S1-N{NN}", f"A-S0-N{NN}", f"A-S1-N{NN}-noclock"}
    assert [c.name for c, _ in cells[32:34]] == [f"B-S1-N{NN}", f"C-S1-N{NN}"] and cells[32][1] == 0
    assert [s for _, s in cells[32:48]] == [s for s in range(8) for _ in range(2)]


def test_pilot_subset_lands_in_the_grid_run_dirs():
    cells = rg.plan("confirmatory", conditions=["A-S1"], seeds=[0, 1, 2])
    assert [(c.name, s) for c, s in cells] == [(f"A-S1-N{NN}", 0), (f"A-S1-N{NN}", 1), (f"A-S1-N{NN}", 2)]
    assert rg.plan("confirmatory", conditions=[f"A-S1-N{NN}-noclock"], seeds=[3])[0][0].name == f"A-S1-N{NN}-noclock"
    run_dir = rg.run_directory(Path("runs"), cells[0][0].config, 0)
    assert run_dir == Path("runs") / f"A-S1-N{NN}" / "seed0"
    with pytest.raises(ValueError):
        rg.plan("minimum", conditions=["B-S1"])
    with pytest.raises(ValueError):
        rg.plan("confirmatory", seeds=[9])
    assert len(rg.plan("full", seeds=[8, 9])) == 12


def test_parse_seeds_forms():
    assert rg.parse_seeds("0-2") == [0, 1, 2]
    assert rg.parse_seeds("0,3,5") == [0, 3, 5]
    assert rg.parse_seeds("0 1 2") == [0, 1, 2]
    assert rg.parse_seeds("0-1 7") == [0, 1, 7]
    assert rg.parse_seeds(None) is None


def test_command_uses_the_fallback_trainer_with_threads_and_budget():
    cond, seed = rg.plan("confirmatory", conditions=["A-S1"], seeds=[0])[0]
    cmd = rg.command(cond, seed, Path("runs"), threads=2, agent_steps=None, extra=[])
    assert cmd[:3] == [sys.executable, "-m", "hamlet.train_fallback"]
    assert "--threads" in cmd and "--agent-steps" not in cmd and "--runners" not in cmd
    smoke = rg.command(cond, seed, Path("runs/_x"), threads=1, agent_steps=100_000, extra=["--no-codecarbon"])
    assert smoke[-3:] == ["--agent-steps", "100000", "--no-codecarbon"]


def test_manifest_and_status_lifecycle(tmp_path):
    cond, seed = rg.plan("confirmatory", conditions=["A-S1"], seeds=[0])[0]
    run_dir = rg.run_directory(tmp_path, cond.config, seed)
    assert rg.cell_status(run_dir) == "pending"
    cmd = rg.command(cond, seed, tmp_path, 2, None, [])
    rg.write_manifest(run_dir, cond, seed, cmd, 2, None, pid=999_999_999)
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["condition"] == f"A-S1-N{NN}" and m["seed"] == 0 and m["policy_seed"] == 100
    budget = rg.PPOHyperparameters().total_agent_steps      # the registered budget, 30M
    assert m["step_budget"] == budget and m["hyperparameters"]["total_agent_steps"] == budget
    assert set(m["git"]) == {"describe", "commit"} and m["status"] == "running"
    assert m["thread_env"] == {k: "2" for k in rg.THREAD_ENV} and m["started_utc"].endswith("Z")
    assert m["config"]["arm"] == "A" and m["command"] == cmd
    # a dead pid without selection.json is a crashed cell: rerun, not skip
    assert rg.cell_status(run_dir) == "crashed"
    # a live pid that is not a trainer of this seed (this test process) is PID reuse, not a running cell
    rg.write_manifest(run_dir, cond, seed, cmd, 2, None, pid=rg.os.getpid())
    assert rg.cell_status(run_dir, seed=seed) == "crashed"
    (run_dir / "selection.json").write_text("{}")
    assert rg.cell_status(run_dir) == "complete"
    # finished at the launch budget: complete; finished at another budget: refused, never skipped
    assert rg.cell_status(run_dir, budget=budget) == "complete"
    assert rg.cell_status(run_dir, budget=100_000) == "budget-mismatch"
    assert rg.recorded_budget(run_dir) == budget
    done = rg.finish_manifest(run_dir, 0, started=0.0)
    assert done["status"] == "complete" and done["selection_json"] is True and done["finished_utc"].endswith("Z")


def test_estimate_scales_with_cells_and_concurrency():
    cells = rg.plan("confirmatory")
    est = rg.estimate(cells, concurrent=4, rate=13_000.0, agent_steps=10_000_000)
    # checkpoints follow the batch, which follows N: 10M / (240 * 16 * N) updates, one checkpoint
    # every 16 of them plus the final one. 21 at N = 8, 19 at N = 9.
    expected_ckpts = math.ceil(10_000_000 / rg.PPOHyperparameters().train_batch_size) // 16 + 1
    assert est["runs"] == 48 and est["checkpoints_per_run"] == expected_ckpts
    assert est["per_run_min"] == pytest.approx(10_000_000 / 13_000 / 60)
    assert est["concurrent_h"] == pytest.approx(est["sequential_h"] / 4)
    assert est["disk_with_eval_mb"] > est["disk_train_mb"] > 0


def test_thread_oversubscription_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(rg.os, "cpu_count", lambda: 8)
    with pytest.raises(SystemExit, match="exceeds"):
        rg.main(["--dry-run", "--concurrent", "5", "--threads", "2"])
    rg.main(["--dry-run", "--conditions", "A-S1", "--seeds", "0", "1", "2", "--concurrent", "3", "--threads", "2"])
    out = capsys.readouterr().out
    assert out.count("hamlet.train_fallback") == 3 and "--seed 2" in out


def test_smoke_budgets_stay_out_of_the_study_root_and_never_log_energy(capsys, tmp_path):
    with pytest.raises(SystemExit, match="smoke budget"):
        rg.main(["--dry-run", "--conditions", "A-S1", "--seeds", "0", "--agent-steps", "1000", "--concurrent", "1"])
    rg.main(["--dry-run", "--conditions", "A-S1", "--seeds", "0", "--agent-steps", "1000", "--concurrent", "1", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "--no-codecarbon" in out and "--agent-steps 1000" in out


def test_study_launch_refuses_a_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "running_trainers", lambda: [])
    monkeypatch.setattr(rg, "git_provenance", lambda: {"describe": "1be4aab-dirty", "commit": "x"})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit, match="committed, tagged tree"):
        rg.main(["--conditions", "A-S1", "--seeds", "0", "--concurrent", "1"])