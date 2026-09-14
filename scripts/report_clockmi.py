"""Validation report for the clock-information statistic and its null.

The statistic is I(zone; hour_bin | state_bin, open_code, night): the
information the hour inside a window carries about the zone once the state
bins and the schedule are known. The default null relabels hour bins within
each schedule stratum (see hamlet.metrics.routine.conditional_mi). This script
produces every number quoted about that null::

    uv run python scripts/report_clockmi.py                # all sections
    uv run python scripts/report_clockmi.py --section cells

Sections
  cells      occupied (cond, hour, zone) and (cond, hour) cells, data vs null mean, per baseline
             policy and relabel scope; a null that occupies more cells than the data carries more
             plug-in bias and reads negative on a clock-locked policy
  synthetic  the schedule-only process (no within-window clock use by construction) over many
             seeds: mean, sd, min, max of z per null and relabel scope, plus the planted variant
  bins       the baseline policies and the planted real-log dependence at tercile and quintile
             state bins
  plant      the planted dependence in words and numbers: which rows moved, by how much, and
             the z it produces, so the detectable scale of the metric is on record
  gain       the H1 statistic (held-out predictive gain, bits per tick) on the baseline policies,
             the planted dependence, the synthetic process over twelve seeds, at terciles (primary)
             and at the two robustness binnings, quintiles and fixed absolute edges (1/3, 2/3)
  markov     a rejected null kept for the record: resimulating the zone from a first-order model
             along the real state and clock columns; matches the (state, hour) cells exactly and
             over-occupies the (state, hour, zone) cells
  chronotype the H1 positive control: the chronotype_split population (half the agents with the
             bedtime window two hours early, half two hours late) under GREEDY-CLOCK on the
             evaluation seeds, its predictive gain and z beside neutral GREEDY-CLOCK's at terciles
             and quintiles, and the per-seed contrast against the registered H1a threshold
  statemi    the H1b statistic, I(zone; state_bin | hour_bin, open_code, night), on the baseline
             policies under its block null (the state-bin sequence of each agent-day cyclically
             shifted, zone and clock fixed) and, as diagnostics, under the relabel and tick nulls,
             with the state bins of the next agent in place of the agent's own (a pairing control
             that keeps every marginal and the global phase of the states but removes the
             coupling between an agent's own zone history and its own states), and the held-out
             predictive gain of the state bins beyond the clock

Inputs: evaluation logs under runs/<condition>/<policy>/seed<seed>_ep0.parquet,
where the condition name comes from HamletConfig (A-S1-N9)
(from scripts/run_baselines.py). Outputs: markdown on stdout and CSV copies
under runs/reports/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_constants as K
from hamlet.config import SOCIAL, HamletConfig
from hamlet.metrics.routine import add_bins, conditional_mi, predictive_gain
from hamlet.metrics.synthetic import PLANT_BIN, PLANT_PROB, schedule_only_log

OUT_DIR = Path("runs/reports")
POLICIES = ("RANDOM", "GREEDY-STATE", "GREEDY-CLOCK", "GREEDY-CLOCK-WORK")
SEEDS = (10000, 10001, 10002)
SCOPES = ("agent_day", "agent", "day")
SYNTH_SEEDS = tuple(range(31, 43))          # twelve seeds
N_PERM = 150


BINNINGS = ((3, "quantile"), (5, "quantile"), (3, "fixed"))   # primary, then the two robustness variants


def binning_label(n_state_bins: int, state_edges: str) -> str:
    return f"{n_state_bins} {state_edges}"


def load(policy: str, seed: int, n_state_bins: int = 3, state_edges: str = "quantile") -> pd.DataFrame:
    """One evaluation episode with bins; quantile cut points come from the policy's pooled episodes,
    the same convention as scripts/run_baselines.py, so the two scripts agree to the digit; fixed
    edges (1/k, ..., (k-1)/k of [0, 1]) ignore the reference."""
    cfg = HamletConfig()
    folder = Path(f"runs/{HamletConfig().condition_name}/{policy}")
    pooled = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob("seed*_ep*.parquet"))], ignore_index=True)
    df = pd.read_parquet(folder / f"seed{seed}_ep0.parquet")
    return add_bins(df, ref=pooled, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night,
                    n_state_bins=n_state_bins, state_edges=state_edges)


def plant(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Force zone = SOCIAL during PLANT_BIN inside the job window, per agent-day with PLANT_PROB.

    Returns the modified frame and the fraction of rows changed.
    """
    cfg = HamletConfig()
    out = df.copy()
    in_bin = (out["t_day"] >= PLANT_BIN[0]) & (out["t_day"] < PLANT_BIN[1]) & (out["open_code"] == 3)
    keys = out.loc[in_bin, ["agent", "day"]].drop_duplicates()
    chosen = keys[rng.random(len(keys)) < PLANT_PROB]
    mask = in_bin & out.set_index(["agent", "day"]).index.isin(chosen.set_index(["agent", "day"]).index)
    changed = float((mask & (out["zone"] != SOCIAL)).mean())
    out.loc[mask, "zone"] = SOCIAL
    return out, changed


# ---- sections ---------------------------------------------------------------------
def section_cells(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pol in POLICIES:
        for seed in SEEDS:
            df = load(pol, seed)
            for scope in SCOPES:
                r = conditional_mi(df, n_perm=N_PERM, rng=rng, relabel_scope=scope, report_cells=True)
                rows.append({
                    "policy": pol, "seed": seed, "scope": scope, "z": r["z"],
                    "cells_cgt_real": r["cells_cgt_real"], "cells_cgt_null": r["cells_cgt_null"],
                    "cgt_ratio": r["cells_cgt_null"] / r["cells_cgt_real"],
                    "cells_cg_real": r["cells_cg_real"], "cells_cg_null": r["cells_cg_null"],
                    "cg_ratio": r["cells_cg_null"] / r["cells_cg_real"],
                })
    return pd.DataFrame(rows)


def section_synthetic(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    variants = [("default", {}), ("slow integrator", {"leak": 0.003}), ("sticky", {"stick": 0.97})]
    for name, kw in variants:
        for seed in SYNTH_SEEDS:
            clean = schedule_only_log(seed, **kw)
            planted = schedule_only_log(seed, plant=SOCIAL, **kw)
            for scope in SCOPES:
                z = conditional_mi(clean, n_perm=N_PERM, rng=rng, relabel_scope=scope)["z"]
                zp = conditional_mi(planted, n_perm=N_PERM, rng=rng, relabel_scope=scope)["z"]
                rows.append({"variant": name, "seed": seed, "null": f"relabel/{scope}", "z_clean": z, "z_planted": zp})
            rows.append({"variant": name, "seed": seed, "null": "block",
                         "z_clean": conditional_mi(clean, n_perm=N_PERM, rng=rng, null="block")["z"],
                         "z_planted": conditional_mi(planted, n_perm=N_PERM, rng=rng, null="block")["z"]})
    df = pd.DataFrame(rows)
    summary = df.groupby(["variant", "null"], sort=False).agg(
        z_clean_mean=("z_clean", "mean"), z_clean_sd=("z_clean", "std"), z_clean_min=("z_clean", "min"),
        z_clean_max=("z_clean", "max"), z_planted_mean=("z_planted", "mean"), z_planted_min=("z_planted", "min"),
    ).reset_index()
    return summary


def section_bins(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for k in (3, 5):
        for pol in POLICIES:
            for seed in SEEDS:
                df = load(pol, seed, n_state_bins=k)
                z = conditional_mi(df, n_perm=N_PERM, rng=rng)["z"]
                rows.append({"state_bins": k, "policy": pol, "seed": seed, "z": z})
            if pol == "GREEDY-CLOCK":
                for seed in SEEDS:
                    df, _ = plant(load(pol, seed, n_state_bins=k), rng)
                    rows.append({"state_bins": k, "policy": "planted on GREEDY-CLOCK", "seed": seed,
                                 "z": conditional_mi(df, n_perm=N_PERM, rng=rng)["z"]})
    return pd.DataFrame(rows)


def section_plant(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        base = load("GREEDY-CLOCK", seed)
        planted, changed = plant(base, rng)
        r0 = conditional_mi(base, n_perm=N_PERM, rng=rng)
        r1 = conditional_mi(planted, n_perm=N_PERM, rng=rng)
        rows.append({"seed": seed, "rows_changed_frac": changed, "value_before_bits": r0["value"],
                     "value_after_bits": r1["value"], "z_before": r0["z"], "z_after": r1["z"]})
    return pd.DataFrame(rows)


def markov_null_draw(zone: np.ndarray, state: np.ndarray, sched: np.ndarray, agent: np.ndarray, day: np.ndarray,
                     tick: np.ndarray, rng: np.random.Generator, n_zones: int, alpha: float = 0.5) -> np.ndarray:
    """Resimulate the zone along the real state and clock columns from a first-order model.

    A rejected null, kept so its failure is reproducible: the model is fitted
    per (schedule stratum, state bin, previous zone) and the zone sequence is
    regenerated block by agent-day. The (state, hour) cells are then identical
    to the data by construction, but the simulated zones occupy far more
    (state, hour, zone) cells than a deterministic scheduler does.
    """
    n = len(zone)
    order = np.lexsort((tick, agent))
    z, st, sc, a, d = zone[order], state[order], sched[order], agent[order], day[order]
    key = sc * (st.max() + 1) + st
    n_key = int(key.max()) + 1
    new_block = np.r_[True, (a[1:] != a[:-1]) | (d[1:] != d[:-1])]
    prev = np.where(new_block, n_zones, np.r_[n_zones, z[:-1]])
    counts = np.zeros((n_key, n_zones + 1, n_zones))
    np.add.at(counts, (key, prev, z), 1)
    cum = ((counts + alpha) / (counts + alpha).sum(axis=2, keepdims=True)).cumsum(axis=2)
    starts = np.flatnonzero(new_block)
    lengths = np.diff(np.r_[starts, n])
    out = np.empty(n, dtype=np.int64)
    prev_z = np.full(len(starts), n_zones)
    u = rng.random(n)
    for pos in range(int(lengths.max())):
        alive = lengths > pos
        idx = starts[alive] + pos
        choice = (u[idx][:, None] > cum[key[idx], prev_z[alive]]).sum(axis=1)
        out[idx] = choice
        prev_z[alive] = choice
    res = np.empty(n, dtype=np.int64)
    res[order] = out
    return res


def section_markov(rng: np.random.Generator) -> pd.DataFrame:
    from hamlet.metrics.common import group_codes
    from hamlet.metrics.routine import _occupied_cells, _plugin_cmi_bits
    from hamlet.config import ZONE_NAMES

    rows = []
    for pol in POLICIES:
        for seed in SEEDS:
            df = load(pol, seed)
            df = df[df["day"] >= HamletConfig().burn_in_days].reset_index(drop=True)
            t = group_codes(df, ("zone",))
            g = group_codes(df, ("hour_bin",))
            c = group_codes(df, ("state_bin", "open_code", "night"))
            st = group_codes(df, ("state_bin",))
            sc = group_codes(df, ("open_code", "night"))
            real = _plugin_cmi_bits(t, g, c)
            real_cells = _occupied_cells(t, g, c)
            nulls, cells = [], []
            for _ in range(60):
                tn = markov_null_draw(t, st, sc, df["agent"].to_numpy(), df["day"].to_numpy(), df["t"].to_numpy(),
                                      rng, len(ZONE_NAMES))
                nulls.append(_plugin_cmi_bits(tn, g, c))
                cells.append(_occupied_cells(tn, g, c))
            nulls = np.array(nulls)
            cells = np.array(cells)
            rows.append({"policy": pol, "seed": seed, "z": (real - nulls.mean()) / nulls.std(),
                         "cgt_ratio": cells[:, 0].mean() / real_cells[0], "cg_ratio": cells[:, 1].mean() / real_cells[1]})
    return pd.DataFrame(rows)


STATE_GIVEN = ("state_bin",)
STATE_COND = ("hour_bin", "open_code", "night")


def cross_agent_states(df: pd.DataFrame) -> pd.DataFrame:
    """The same log with every agent's state bins replaced by the next agent's at the same tick.

    A pairing control for the state-MI: each agent keeps its own zone
    sequence and the population keeps every state marginal and the shared
    phase of the states along the clock, but the coupling between an agent's
    own zone history and its own states (a visit to the canteen raises that
    agent's satiety) is removed. A statistic that reads only "zone follows
    state" should fall to its null on a policy that never reads its states.
    """
    out = df.sort_values(["t", "agent"]).reset_index(drop=True)
    n = int(out["agent"].max()) + 1
    shifted = out["state_bin"].to_numpy().reshape(-1, n)[:, np.r_[1:n, 0]].reshape(-1)
    out["state_bin"] = shifted
    return out


def section_statemi(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pol in POLICIES:
        for seed in SEEDS:
            df = load(pol, seed)
            block = conditional_mi(df, given=STATE_GIVEN, cond=STATE_COND, n_perm=N_PERM, rng=rng, null="block")
            relabel = conditional_mi(df, given=STATE_GIVEN, cond=STATE_COND, n_perm=N_PERM, rng=rng, null="relabel")
            tick = conditional_mi(df, given=STATE_GIVEN, cond=STATE_COND, n_perm=N_PERM, rng=rng, null="tick")
            crossed = conditional_mi(cross_agent_states(df), given=STATE_GIVEN, cond=STATE_COND, n_perm=N_PERM, rng=rng, null="block")
            gain = predictive_gain(df, given=STATE_GIVEN, cond=STATE_COND, n_perm=N_PERM, rng=rng)
            rows.append({
                "policy": pol, "seed": seed, "value_bits": block["value"],
                "block_null_mean": block["null_mean"], "block_z": block["z"],
                "corrected_bits": block["value"] - block["null_mean"],
                "relabel_z": relabel["z"], "tick_z": tick["z"],
                "cross_agent_value_bits": crossed["value"], "cross_agent_block_z": crossed["z"],
                "gain_bits": gain["value"], "gain_z": gain["z"],
            })
    return pd.DataFrame(rows)


CONTROL_POPULATION = "chronotype_split"
CONTROL_TRAITS_SEED = 0       # one trait assignment for every evaluation seed, as in report_gate.py
# The registered threshold, read from the one place that holds it rather than copied, so the
# chronotype contrast is always printed against the current registered value.
H1_THRESHOLD = K.H1A_THRESHOLD.value


def control_logs(population: str, seeds: tuple[int, ...], traits_seed: int) -> dict[int, pd.DataFrame]:
    """One GREEDY-CLOCK episode per evaluation seed on ``population``, with the evaluate.py conventions."""
    import dataclasses
    from hamlet.evaluate import rollout
    from hamlet.policies import GreedyClockPolicy

    cfg = dataclasses.replace(HamletConfig(), population=population)
    return {seed: rollout(cfg, GreedyClockPolicy(), seed, 0, traits_seed=traits_seed) for seed in seeds}


def section_chronotype(rng: np.random.Generator) -> pd.DataFrame:
    cfg = HamletConfig()
    logs = control_logs(CONTROL_POPULATION, SEEDS, CONTROL_TRAITS_SEED)
    pooled = pd.concat(logs.values(), ignore_index=True)
    rows = []
    for k, edges in BINNINGS:
        ref = {}
        for seed in SEEDS:
            r = predictive_gain(load("GREEDY-CLOCK", seed, n_state_bins=k, state_edges=edges), n_perm=N_PERM, rng=rng)
            ref[seed] = r
        for seed in SEEDS:
            df = add_bins(logs[seed], ref=pooled, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night,
                          n_state_bins=k, state_edges=edges)
            r = predictive_gain(df, n_perm=N_PERM, rng=rng)
            d = r["value"] - ref[seed]["value"]
            rows.append({"state_bins": binning_label(k, edges), "seed": seed,
                         "gain_control": r["value"], "z_control": r["z"],
                         "gain_greedy_clock": ref[seed]["value"], "z_greedy_clock": ref[seed]["z"],
                         "contrast_bits": d, "clears_threshold": "yes" if d >= H1_THRESHOLD else "NO"})
    return pd.DataFrame(rows)


def _gain_rows(rng: np.random.Generator, binnings) -> list[dict]:
    rows = []
    for k, edges in binnings:
        for pol in POLICIES:
            for seed in SEEDS:
                r = predictive_gain(load(pol, seed, n_state_bins=k, state_edges=edges), n_perm=N_PERM, rng=rng)
                rows.append({"state_bins": binning_label(k, edges), "policy": pol, "seed": seed, "gain_bits": r["value"], "z": r["z"]})
        for seed in SEEDS:
            df, _ = plant(load("GREEDY-CLOCK", seed, n_state_bins=k, state_edges=edges), rng)
            r = predictive_gain(df, n_perm=N_PERM, rng=rng)
            rows.append({"state_bins": binning_label(k, edges), "policy": "planted on GREEDY-CLOCK", "seed": seed, "gain_bits": r["value"], "z": r["z"]})
    return rows


def section_gain(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    # draw order: quantile binnings, then the synthetic process, then the fixed-edge binning (added
    # later), so the numbers recorded from the earlier layout of this section are unchanged
    rows = _gain_rows(rng, [b for b in BINNINGS if b[1] == "quantile"])
    srows = []
    variants = [("default", {}), ("slow integrator", {"leak": 0.003}), ("sticky", {"stick": 0.97})]
    for name, kw in variants:
        for seed in SYNTH_SEEDS:
            clean = predictive_gain(add_bins(schedule_only_log(seed, **kw)), n_perm=80, rng=rng)["z"]
            planted = predictive_gain(add_bins(schedule_only_log(seed, plant=SOCIAL, **kw)), n_perm=80, rng=rng)["z"]
            srows.append({"variant": name, "seed": seed, "z_clean": clean, "z_planted": planted})
    synth = pd.DataFrame(srows).groupby("variant", sort=False).agg(
        z_clean_mean=("z_clean", "mean"), z_clean_sd=("z_clean", "std"), z_clean_min=("z_clean", "min"),
        z_clean_max=("z_clean", "max"), z_planted_mean=("z_planted", "mean"), z_planted_min=("z_planted", "min"),
    ).reset_index()
    rows += _gain_rows(rng, [b for b in BINNINGS if b[1] != "quantile"])
    real = pd.DataFrame(rows)
    return real, synth


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=["all", "cells", "synthetic", "bins", "plant", "gain", "markov", "statemi", "chronotype"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def section_rng(index: int) -> np.random.Generator:
        """One stream per section (plant 1, cells 2, synthetic 3, gain 4, markov 5, bins 6, statemi 7, chronotype 8),
        so a section's numbers do not depend on which others ran."""
        return np.random.default_rng([args.seed, index])
    pd.set_option("display.width", 220)

    def emit(name: str, df: pd.DataFrame) -> None:
        print(f"## {name}\n{df.round(3).to_markdown(index=False)}\n")
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    print("statistic: I(zone; hour_bin | state_bin, open_code, night); default null: relabel within "
          "(agent, day, open_code, night); hour bins of 30 ticks; state bins are equal-frequency quantiles\n")
    if args.section in ("all", "plant"):
        print(f"planted dependence: inside the job window, during ticks {PLANT_BIN[0]}-{PLANT_BIN[1]} "
              f"(12:00-15:00, one hour bin), every agent-day is forced to SOCIAL for the whole bin with "
              f"probability {PLANT_PROB}; nothing else moves.\n")
        emit("clockmi_plant", section_plant(section_rng(1)))
    if args.section in ("all", "cells"):
        df = section_cells(section_rng(2))
        emit("clockmi_cells", df)
        emit("clockmi_cells_mean", df.drop(columns="seed").groupby(["policy", "scope"], sort=False).mean().reset_index())
    if args.section in ("all", "synthetic"):
        emit("clockmi_synthetic", section_synthetic(section_rng(3)))
    if args.section in ("all", "gain"):
        real, synth = section_gain(section_rng(4))
        emit("gain_real", real)
        emit("gain_real_mean", real.drop(columns="seed").groupby(["state_bins", "policy"], sort=False).agg(["mean", "min", "max"]).reset_index())
        emit("gain_synthetic", synth)
    if args.section in ("all", "markov"):
        df = section_markov(section_rng(5))
        emit("clockmi_markov", df)
        emit("clockmi_markov_mean", df.drop(columns="seed").groupby("policy", sort=False).mean().reset_index())
    if args.section in ("all", "chronotype"):
        print(f"positive control for H1: population {CONTROL_POPULATION} under GREEDY-CLOCK, trait seed "
              f"{CONTROL_TRAITS_SEED}, tercile cut points pooled over its own episodes; contrast = control gain "
              f"minus neutral GREEDY-CLOCK gain on the same evaluation seed, against {H1_THRESHOLD} bits per tick.\n")
        emit("chronotype_control", section_chronotype(section_rng(8)))
    if args.section in ("all", "statemi"):
        df = section_statemi(section_rng(7))
        emit("statemi", df)
        emit("statemi_mean", df.drop(columns="seed").groupby("policy", sort=False).agg(["mean", "min", "max"]).reset_index())
    if args.section in ("all", "bins"):
        df = section_bins(section_rng(6))
        emit("clockmi_bins", df)
        emit("clockmi_bins_mean", df.drop(columns="seed").groupby(["state_bins", "policy"], sort=False)["z"].agg(["mean", "min", "max"]).reset_index())


if __name__ == "__main__":
    main()
