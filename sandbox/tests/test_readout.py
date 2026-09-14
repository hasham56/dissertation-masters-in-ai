"""Goal readout on a synthetic log with known windows, and read-only on a confirmatory baseline Parquet."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hamlet.config import CANTEEN, FARM, HOME, SOCIAL, TRANSIT
from hamlet.metrics.synthetic import make_log

from sandbox import RUNS_DIR
from sandbox.readout import infer_windows, modal_zone_per_hour_bin, read_goals, schedule_sentence, tick_to_clock, write_goals

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "runs" / "A-S1-N8" / "GREEDY-CLOCK" / "seed10000_ep0.parquet"
TEST_DIR = RUNS_DIR / "_test_readout"
T, N, DAYS = 240, 8, 4


def _routine_log():
    """HOME 0-60, walk 60-80, FARM 80-180 (active), CANTEEN 184-196 (active), SOCIAL 200-220 (active), HOME 220-240."""
    rng = np.random.default_rng(0)
    base = make_log(rng, n_agents=N, n_days=DAYS, ticks_per_day=T)
    t = base["t_day"].to_numpy()
    zone = np.select([t < 60, t < 80, t < 180, t < 184, t < 196, t < 200, t < 220],
                     [HOME, TRANSIT, FARM, TRANSIT, CANTEEN, TRANSIT, SOCIAL], default=HOME)
    active = np.isin(zone, [HOME, FARM, CANTEEN, SOCIAL])
    return make_log(np.random.default_rng(0), n_agents=N, n_days=DAYS, ticks_per_day=T, zone=zone, active=active)


def test_tick_to_clock():
    assert tick_to_clock(0) == "00:00" and tick_to_clock(65) == "06:30" and tick_to_clock(239) == "23:54"


def test_windows_and_sentence_on_a_known_routine():
    df = _routine_log()
    sub = df[(df["agent"] == 0) & (df["day"] >= 1)]
    w = infer_windows(sub)
    assert w["wake"]["tick"] == 60 and w["wake"]["clock"] == "06:00"
    assert w["sleep"]["tick"] == 220 and w["sleep"]["clock"] == "22:00"
    assert w["work"]["start_tick"] == 80 and w["work"]["end_tick"] == 180 and w["work"]["zone"] == "FARM"
    assert len(w["meals"]) == 1 and w["meals"][0]["tick"] == 184
    assert w["social"]["start_tick"] == 200 and w["social"]["end_tick"] == 220
    sentence = schedule_sentence(0, w)
    assert sentence == ("Agent 0 is up at 06:00, works at the farm 08:00 to 18:00, eats around 18:24, "
                        "socialises 20:00 to 22:00 and is home for the night by 22:00.")
    bins = modal_zone_per_hour_bin(sub)
    # 06:00-09:00 is twenty ticks of walking against ten of farm; 18:00-21:00 twelve canteen ticks against ten social
    assert [b["zone"] for b in bins] == ["HOME", "HOME", "TRANSIT", "FARM", "FARM", "FARM", "CANTEEN", "HOME"]
    assert bins[0]["share"] == pytest.approx(1.0)
    goals = read_goals(df, burn_in_days=1)
    assert set(goals["agents"]) == {str(k) for k in range(N)} and goals["exploratory"] is True
    assert goals["agents"]["3"]["schedule"].startswith("Agent 3 is up at 06:00")


def test_absent_windows_are_left_out_of_the_sentence():
    rng = np.random.default_rng(1)
    df = make_log(rng, n_agents=2, n_days=3, ticks_per_day=T, zone=np.full(2 * 3 * T, TRANSIT), active=np.zeros(2 * 3 * T, bool))
    w = infer_windows(df[df["agent"] == 0])
    assert w["work"] is None and w["social"] is None and w["sleep"] is None and w["meals"] == []
    assert w["wake"]["tick"] == 0
    assert schedule_sentence(0, w) == "Agent 0 is up at 00:00."


@pytest.mark.skipif(not BASELINE.exists(), reason="no baseline Parquet under runs/ on this machine")
def test_readout_runs_read_only_on_a_confirmatory_baseline_parquet():
    before = (BASELINE.stat().st_mtime_ns, hashlib.sha256(BASELINE.read_bytes()).hexdigest())
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    out = write_goals(BASELINE, TEST_DIR / "goals.json")
    after = (BASELINE.stat().st_mtime_ns, hashlib.sha256(BASELINE.read_bytes()).hexdigest())
    assert before == after
    assert not any(p.name.startswith("goals") for p in BASELINE.parent.iterdir())
    goals = json.loads(out.read_text())
    assert goals["source"]["condition"] == "A-S1-N8" and goals["source"]["seed"] == 10_000
    assert len(goals["agents"]) == 8
    # GREEDY-CLOCK rests at night under the hysteresis rule: home for the night on every agent, up around 06:00
    for entry in goals["agents"].values():
        assert len(entry["modal_zone_per_hour_bin"]) == 8
        assert entry["schedule"].startswith("Agent ")
        assert entry["windows"]["sleep"] is not None and 180 <= entry["windows"]["sleep"]["tick"] <= 239
        assert entry["windows"]["wake"] is not None and 50 <= entry["windows"]["wake"]["tick"] <= 90
        assert entry["windows"]["work"] is None          # the scheduler never works
    with pytest.raises(ValueError):
        write_goals(BASELINE, BASELINE.parent / "goals.json")
