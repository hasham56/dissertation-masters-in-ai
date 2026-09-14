"""The per-seed table merge and the section cache of scripts/compute_metrics.py, on synthetic logs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hamlet.config import HamletConfig
from hamlet.metrics.synthetic import make_log

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import compute_metrics as cm  # noqa: E402

# The condition name carries the population size, so the tests follow it rather than
# pinning a literal.
NN = HamletConfig().n_agents


def test_merge_per_seed_replaces_rows_and_never_duplicates():
    old = pd.DataFrame({"condition": [f"A-S1-N{NN}"] * 3, "label": ["ckpt_0320", "ckpt_0320", "GREEDY-CLOCK"],
                        "seed": [0, 1, "pooled"], "pass": ["stochastic"] * 3, "x": [1.0, 2.0, 3.0]})
    new = pd.DataFrame({"condition": [f"A-S1-N{NN}", f"A-S1-N{NN}"], "label": ["ckpt_0320", "GREEDY-CLOCK"],
                        "seed": [1, "pooled"], "pass": ["stochastic"] * 2, "x": [20.0, 30.0]})
    merged = cm.merge_per_seed(old, new)
    assert len(merged) == 3
    assert merged.set_index(["label", "seed"])["x"].to_dict() == {("ckpt_0320", 0): 1.0, ("ckpt_0320", 1): 20.0, ("GREEDY-CLOCK", "pooled"): 30.0}
    # a CSV round trip turns ints into strings mixed with "pooled": still no duplicates
    text = merged.to_csv(index=False)
    reread = pd.read_csv(pd.io.common.StringIO(text))
    again = cm.merge_per_seed(reread, new)
    assert len(again) == 3


def _episode_files(tmp_path: Path, n: int) -> list[Path]:
    files = []
    for k in range(n):
        rng = np.random.default_rng(k)
        df = make_log(rng, n_agents=8, n_days=6, ticks_per_day=240)
        df["seed"] = 10_000 + k
        path = tmp_path / f"seed{10_000 + k}_ep0.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        files.append(path)
    return files


def test_partial_section_rerun_keeps_the_other_columns(tmp_path):
    files = _episode_files(tmp_path, 2)
    cfg = json.loads(HamletConfig().to_json())
    unit = {"condition": f"A-S1-N{NN}", "label": "test", "seed": 0, "seed_key": 0, "pass": "stochastic",
            "files": [str(f) for f in files], "ref_files": [str(f) for f in files], "ref_scope": "run",
            "metrics_dir": str(tmp_path / "metrics"), "run_dir": None, "config": cfg,
            "sections": ["dol", "dashboard"], "n_perm": 5, "force": False, "replay_path": None}
    first = cm.compute_unit(dict(unit))
    assert "dol_z_dayshuffle" in first and "mean_D" in first
    assert (tmp_path / "metrics" / "dol.json").exists() and (tmp_path / "metrics" / "dashboard.json").exists()
    second = cm.compute_unit({**unit, "sections": ["dashboard"]})
    assert "dol_z_dayshuffle" in second and second["dol_z_dayshuffle"] == first["dol_z_dayshuffle"]
    assert second["_timings"] == {}          # everything came from the cache
    third = cm.compute_unit({**unit, "sections": ["dol"], "force": True})
    assert "mean_D" in third and third["_timings"].keys() == {"dol"}


def test_gain_cache_is_invalidated_when_the_reference_changes(tmp_path):
    files = _episode_files(tmp_path, 2)
    cfg = json.loads(HamletConfig().to_json())
    base = {"condition": f"A-S1-N{NN}", "label": "test", "seed": 0, "seed_key": 0, "pass": "stochastic",
            "files": [str(f) for f in files], "ref_files": [str(files[0])], "ref_scope": "run",
            "metrics_dir": str(tmp_path / "metrics"), "run_dir": None, "config": cfg,
            "sections": ["gain"], "n_perm": 3, "force": False, "replay_path": None}
    first = cm.compute_unit(dict(base))
    assert "gain" in first["_timings"] and "edge_terciles_E_1" in first
    cached = cm.compute_unit(dict(base))
    assert cached["_timings"] == {}
    wider = cm.compute_unit({**base, "ref_files": [str(f) for f in files]})
    assert "gain" in wider["_timings"]        # a different reference recomputes instead of reusing the cache
    assert wider["edge_terciles_E_1"] != first["edge_terciles_E_1"]