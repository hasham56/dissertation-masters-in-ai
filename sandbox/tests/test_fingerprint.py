"""Fingerprint on a synthetic Parquet with known occupancy, and the diff against a shifted copy."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from hamlet.config import FARM, HOME, SOCIAL, ZONE_NAMES
from hamlet.metrics.synthetic import make_log

from sandbox import RUNS_DIR
from sandbox.fingerprint import actogram_png, fingerprint, fingerprint_diff, format_diff, side_by_side_actograms, write_fingerprint

TEST_DIR = RUNS_DIR / "_test_fingerprint"
T, N, DAYS = 240, 8, 3


def _known_log(shift: int = 0) -> pd.DataFrame:
    """HOME for six hours, FARM for twelve, SOCIAL for six, every agent, every day; ``shift`` rotates the day."""
    rng = np.random.default_rng(0)
    base = make_log(rng, n_agents=N, n_days=DAYS, ticks_per_day=T)
    t_day = (base["t_day"].to_numpy() + shift) % T
    zone = np.where(t_day < 60, HOME, np.where(t_day < 180, FARM, SOCIAL))
    states = np.tile(np.array([0.8, 0.6, 0.9]), (len(base), 1))
    return make_log(np.random.default_rng(0), n_agents=N, n_days=DAYS, ticks_per_day=T, zone=zone, states=states)


def test_fingerprint_recovers_known_occupancy():
    df = _known_log()
    fp = fingerprint(df, burn_in_days=1, n_perm=20)
    zf = fp["zone_fraction"]
    assert zf["HOME"] == pytest.approx(0.25) and zf["FARM"] == pytest.approx(0.5) and zf["SOCIAL"] == pytest.approx(0.25)
    assert sum(zf.values()) == pytest.approx(1.0)
    hz = np.asarray(fp["hour_zone"])
    assert hz.shape == (8, len(ZONE_NAMES)) and hz.sum() == pytest.approx(1.0)
    # bins 0-1 are HOME, 2-5 FARM, 6-7 SOCIAL, each bin holding one eighth of the ticks
    for k in range(8):
        expected = HOME if k < 2 else FARM if k < 6 else SOCIAL
        assert hz[k, expected] == pytest.approx(1 / 8)
    assert fp["mean_state"] == pytest.approx({"energy": 0.8, "satiety": 0.6, "social": 0.9})
    assert fp["mean_drive_per_need"] == pytest.approx({"energy": 0.04, "satiety": 0.16, "social": 0.01})
    assert fp["mean_drive"] == pytest.approx(0.21, abs=1e-6)
    # identical agents: no division of labour
    assert fp["dol_indiv"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert fp["days"] == DAYS - 1 and fp["n_agents"] == N
    assert fp["source"]["exploratory"] is True
    json.dumps(fp)


def test_fingerprint_diff_and_pngs():
    a = fingerprint(_known_log(), burn_in_days=1, n_perm=20)
    b = fingerprint(_known_log(shift=60), burn_in_days=1, n_perm=20)
    diff = fingerprint_diff(a, b)
    # a whole-day rotation keeps the zone fractions and moves the hour x zone matrix
    assert all(abs(v) < 1e-9 for v in diff["zone_fraction"].values())
    assert diff["hour_zone_max_abs"] == pytest.approx(1 / 8)
    table = format_diff(a, b, diff)
    assert "zone_fraction[FARM]" in table and "hour_zone max |delta|" in table
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    png = actogram_png(_known_log(), 0, TEST_DIR / "actogram.png")
    both = side_by_side_actograms(_known_log(), _known_log(shift=60), 0, TEST_DIR / "both.png")
    assert png.exists() and both.exists() and png.stat().st_size > 1000


def test_write_fingerprint_reads_a_parquet_and_refuses_outside_paths(tmp_path):
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    src = TEST_DIR / "known.parquet"
    src.parent.mkdir(parents=True, exist_ok=True)
    _known_log().to_parquet(src, engine="pyarrow", index=False)
    json_path, png_path = write_fingerprint(src, TEST_DIR / "out", agent=2)
    assert json_path.exists() and png_path.exists()
    assert json.loads(json_path.read_text())["source"]["path"] == str(src)
    with pytest.raises(ValueError):
        write_fingerprint(src, tmp_path)
