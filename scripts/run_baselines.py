"""Run the hand-written baselines on the evaluation seeds and summarise them.

RANDOM, GREEDY-STATE, GREEDY-CLOCK and GREEDY-CLOCK-WORK are rolled out on
the 32 evaluation seeds of one condition (arm A, symmetry S1, eight agents by default). The
Parquet logs go to ``runs/<condition>/<policy>/`` and one CSV,
``runs/<condition>/baseline_summary.csv``, holds one row per (policy, seed):
the dashboard summary (time-budget fractions, mean states and drive after
burn-in, contention, gossip, return) plus the division-of-labour index, the
clock information beyond the schedule, the state information beyond the clock,
regularity and the null-thresholded co-presence density. The division of labour is reported against two nulls
side by side, the pooled-multinomial and the agent-day shuffle
(``dol_z_multinomial``, ``dol_z_dayshuffle``); the clock information against
three, the tick shuffle, the agent-day clock shift and the within-stratum
relabelling of hour bins (``clockmi_z_tick``, ``clockmi_z_block``,
``clockmi_z_relabel``). The day shuffle and the relabelling are the defaults
of the metric functions; the tick-level nulls treat every tick as an
independent draw and the clock shift is biased against any policy that
follows the schedule (GREEDY-CLOCK scores z about -4 under it while reading
nothing but the opening hours and the night). Both are kept for comparison.
The state information (H1b: I(zone; state_bin | hour_bin, open_code,
night)) is reported with its own block null, the one the analysis plan names
for H1b: the state-bin sequence of each agent-day is cyclically shifted by a
random offset while the zone and the clock columns stay (``state_mi_bits``,
``statemi_null_mean_block``, ``statemi_z_block``, ``state_mi_corrected_bits``
= value minus null mean). RANDOM should sit within |z| 2 under it and
GREEDY-STATE far above. ``clock_gain_bits_fixed`` and ``clock_gain_z_fixed``
are the predictive gain at the fixed-edge state binning (absolute edges at
1/3 and 2/3 of [0, 1], the same for every policy and seed), the second
pre-declared robustness variant beside quintiles; terciles stay primary.
``state_gain_bits``, ``state_gain_null_mean`` and ``state_gain_z`` are the
H1b primary statistic: the held-out predictive gain of adding the state bins
to (hour_bin, open_code, night), with the relabel null applied to the state
labels inside each agent-day-hour stratum; the plug-in state-MI under the
block null stays as a diagnostic column.

Tercile cut points for the state bins are taken from the pooled logs of each
policy, so that every run of a condition shares them. Every permutation null uses a
generator seeded with the evaluation seed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamlet.config import SOCIAL, ZONE_NAMES, HamletConfig  # noqa: E402
from hamlet.evaluate import (  # noqa: E402
    BASELINES,
    add_config_arguments,
    config_from_args,
    evaluate,
    evaluation_seeds,
    make_policy,
)
from hamlet.metrics import dashboard, routine, social, specialisation  # noqa: E402

DEFAULT_POLICIES = ("RANDOM", "GREEDY-STATE", "GREEDY-CLOCK", "GREEDY-CLOCK-WORK")
SUMMARY_FILE = "baseline_summary.csv"


def metric_row(df: pd.DataFrame, pooled: pd.DataFrame, cfg: HamletConfig, n_perm: int) -> dict[str, float]:
    """Extra metrics for one episode log; ``pooled`` fixes the state-bin terciles."""
    seed = int(df["seed"].iloc[0])
    rng = np.random.default_rng(seed)
    binned = routine.add_bins(df, ref=pooled, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night)
    per_day = specialisation.time_budget(df, burn_in_days=cfg.burn_in_days, n_agents=cfg.N, per_day=True)
    counts = per_day.sum(axis=1)
    dol_multinomial = specialisation.dol_indiv(per_day, n_perm=n_perm, rng=rng, null="multinomial")
    dol_dayshuffle = specialisation.dol_indiv(per_day, n_perm=n_perm, rng=rng, null="home_shuffle")
    cmi_tick = routine.conditional_mi(binned, n_perm=n_perm, rng=rng, burn_in_days=cfg.burn_in_days, null="tick")
    cmi_block = routine.conditional_mi(binned, n_perm=n_perm, rng=rng, burn_in_days=cfg.burn_in_days, null="block")
    cmi_relabel = routine.conditional_mi(binned, n_perm=n_perm, rng=rng, burn_in_days=cfg.burn_in_days, null="relabel")
    # own stream (default_rng([seed, 1])), so adding this column moved no previously reported null draw
    state_mi = routine.conditional_mi(binned, given=("state_bin",), cond=("hour_bin", "open_code", "night"),
                                      n_perm=n_perm, rng=np.random.default_rng([seed, 1]),
                                      burn_in_days=cfg.burn_in_days, null="block")
    gain = routine.predictive_gain(binned, n_perm=n_perm, rng=rng, burn_in_days=cfg.burn_in_days)
    # H1b primary: held-out predictive gain of the state bins beyond the clock, relabel null of the
    # state labels inside each agent-day-hour stratum; own stream default_rng([seed, 3])
    state_gain = routine.predictive_gain(binned, given=("state_bin",), cond=("hour_bin", "open_code", "night"),
                                         n_perm=n_perm, rng=np.random.default_rng([seed, 3]),
                                         burn_in_days=cfg.burn_in_days)
    # robustness binning with absolute state edges (1/3, 2/3); own stream default_rng([seed, 2])
    binned_fixed = routine.add_bins(df, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night,
                                    state_edges="fixed")
    gain_fixed = routine.predictive_gain(binned_fixed, n_perm=n_perm, rng=np.random.default_rng([seed, 2]),
                                         burn_in_days=cfg.burn_in_days)
    copresence = social.copresence_counts(df, SOCIAL, cfg.N, cfg.burn_in_days)
    null = social.time_shift_null(df, SOCIAL, cfg.N, n_perm, rng, cfg.burn_in_days)
    row = {
        "dol_indiv": dol_dayshuffle["value"],
        "dol_null_mean_multinomial": dol_multinomial["null_mean"],
        "dol_z_multinomial": dol_multinomial["z"],
        "dol_null_mean_dayshuffle": dol_dayshuffle["null_mean"],
        "dol_z_dayshuffle": dol_dayshuffle["z"],
        "specialisation_index": specialisation.specialisation_index(counts),
        "cmi_clock_given_schedule_bits": cmi_relabel["value"],
        "clockmi_null_mean_tick": cmi_tick["null_mean"],
        "clockmi_z_tick": cmi_tick["z"],
        "clockmi_null_mean_block": cmi_block["null_mean"],
        "clockmi_z_block": cmi_block["z"],
        "clockmi_null_mean_relabel": cmi_relabel["null_mean"],
        "clockmi_z_relabel": cmi_relabel["z"],
        "state_mi_bits": state_mi["value"],
        "statemi_null_mean_block": state_mi["null_mean"],
        "statemi_z_block": state_mi["z"],
        "state_mi_corrected_bits": state_mi["value"] - state_mi["null_mean"],
        "state_gain_bits": state_gain["value"],
        "state_gain_null_mean": state_gain["null_mean"],
        "state_gain_z": state_gain["z"],
        "clock_gain_bits": gain["value"],
        "clock_gain_null_mean": gain["null_mean"],
        "clock_gain_z": gain["z"],
        "clock_gain_bits_fixed": gain_fixed["value"],
        "clock_gain_z_fixed": gain_fixed["z"],
        "regularity": routine.regularity(df, cfg.ticks_per_day, burn_in_days=cfg.burn_in_days),
        "copresence_density": social.thresholded_density(copresence, null),
    }
    for z, name in enumerate(ZONE_NAMES):
        row[f"ticks_{name}"] = int(counts[:, z].sum())
    return row


def summarise(run_dir: Path, policies: Sequence[str], cfg: HamletConfig, n_perm: int) -> pd.DataFrame:
    """One row per (policy, seed) from the Parquet files under ``run_dir``."""
    rows = []
    for name in policies:
        files = sorted((run_dir / name).glob("*.parquet"))
        logs = [pd.read_parquet(f) for f in files]
        pooled = pd.concat(logs, ignore_index=True)
        for f, df in zip(files, logs):
            row = dashboard.summary(df, cfg.burn_in_days, traits_file=dashboard.traits_file_for(f)).iloc[0].to_dict()
            row["policy"] = name
            row.update(metric_row(df, pooled, cfg, n_perm))
            rows.append(row)
    out = pd.DataFrame(rows)
    front = ["policy", "seed", "condition"]
    return out[front + [c for c in out.columns if c not in front]]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--n-seeds", type=int, default=32)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES), choices=sorted(BASELINES))
    parser.add_argument("--n-perm", type=int, default=None, help="permutations per null (default from metrics)")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    seeds = evaluation_seeds(args.n_seeds)
    out = Path(args.out)
    n_perm = args.n_perm if args.n_perm is not None else routine.DEFAULTS.n_perm

    start = time.perf_counter()
    for name in args.policies:
        policy, checkpoint = make_policy(name)
        t0 = time.perf_counter()
        paths = evaluate(cfg, policy, seeds, out, checkpoint=checkpoint, overwrite=args.overwrite)
        print(f"{name:13s} {len(paths)} episodes in {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    run_dir = out / cfg.condition_name
    table = summarise(run_dir, args.policies, cfg, n_perm)
    table.to_csv(run_dir / SUMMARY_FILE, index=False)
    print(f"summary: {run_dir / SUMMARY_FILE} ({len(table)} rows, {time.perf_counter() - t0:.1f} s)")
    shown = ["policy", "seed", "mean_D", "frac_HOME", "frac_FARM", "frac_OFFICE", "frac_CANTEEN", "frac_SOCIAL", "frac_TRANSIT",
             "dol_indiv", "dol_z_multinomial", "dol_z_dayshuffle",
             "cmi_clock_given_schedule_bits", "clockmi_z_relabel", "state_gain_bits", "state_gain_z", "statemi_z_block",
             "clock_gain_bits", "clock_gain_z", "regularity",
             "copresence_density", "gossip_fraction_end", "contention_per_agent_day"]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(table[shown].to_string(index=False))
    print(f"total {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
