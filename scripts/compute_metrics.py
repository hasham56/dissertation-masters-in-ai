"""Every metric of the analysis plan with its null, over every run and baseline, into one per-seed table.

    uv run python scripts/compute_metrics.py                                # runs/ -> runs/metrics/per_seed.csv
    uv run python scripts/compute_metrics.py --root runs/_smoke_post --n-eval-seeds 4 --jobs 1
    uv run python scripts/compute_metrics.py --sections gain dol --conditions A-S1-N9

One row per (condition, label, training seed, pass) in ``<root>/metrics/per_seed.csv``, the
table the confirmatory contrasts and figures are read from (``hamlet.metrics.stats.per_seed``); the
per-agent and per-episode values behind each row are kept in
``<run_dir>/metrics/<section>.json`` (baselines: ``<root>/metrics/baselines/<policy>/``).
Resumable per (run, section): a section is skipped when its JSON exists; ``--force`` redoes it.
Every metric is the function that exists in ``hamlet.metrics``; this script only arranges
frames, draws nulls through those functions and reduces to one value per seed.

Sections (``--sections``), each with its own seeded stream ``default_rng([seed, index])`` as
``scripts/report_clockmi.py`` does, so a section rerun alone reproduces the full run:

  gain (1)      H1a clock gain (``routine.predictive_gain``, hour_bin beyond state terciles + schedule) at
                terciles, quintiles and fixed edges; H1b/H2c state gain (state_bin beyond the clock);
                relabel null, 200 draws. The plan's registered reduction is one count table per evaluation
                episode, pooled over the N agents: one value
                per (run, evaluation seed), the row carrying the median over the evaluation seeds and
                ``metrics/per_eval_seed.csv`` carrying the values H1a's d_s needs (the median over
                evaluation seeds of the difference against GREEDY-CLOCK). It reproduces the reference the
                plan registers: GREEDY-CLOCK 0.090 bits per tick at terciles, z 12.18 at N = 9
                (0.031 and z 5.23 at N = 8). ``*_pooled32_*`` is
                the robustness column the plan names (each agent's 32 episodes pooled first; GREEDY-CLOCK
                0.130 at N = 9, 0.058 at N = 8), to which the registered H1a threshold does not apply. ``--agentmedian`` adds an unregistered
                diagnostic, the gain per agent inside an episode: the plan pools agents because a
                per-agent table is too sparse (about 1,200 ticks against up to 864 cells) and calls that
                conservative, and this is how the claim can be checked. Tercile and quintile edges from
                the run's own 32 evaluation episodes (``--ref-scope run``, the registered default;
                ``condition`` pools the whole condition, ``policy`` is used for baselines).
  cmi (2)       the plug-in clock-MI (relabel, block, tick nulls) and state-MI (block null), diagnostics.
  dol (3)       H3a within-episode DOL_indiv with the agent-day shuffle null (per episode, median of the z
                over episodes), the pooled-multinomial z (descriptive), SI, the pooled DOL over the 32
                episodes, the trait-and-home-shuffle z where replay counts exist; H3b cluster labels per
                episode (``cluster_agents`` with the plan's k = 1 floor when the silhouette is below
                0.25), ``ari_across_episodes`` = ``specialisation.ari_across_episodes`` on those labels
                (the registered statistic; its CI is ``ari_across_episodes_ci`` on the same definition)
                and beside it ``ari_across_episodes_k1zero``, where a pair with a one-cluster labelling
                counts as chance (0). Also run for the init_narrow / init_wide passes (H3d).
  social (4)    H2b null-thresholded co-presence density at SOCIAL (time-shift null, 500 draws), mean
                O/E, arrival-phase variance, contention per agent-day, the gossip curve; per episode,
                median over episodes (mean for the gossip curve).
  routine (5)   secondary: regularity (one-tick slots) and periodogram power at 1/240 for SOCIAL and HOME
                with the per-day circular-shift band of the per-seed statistic (every draw shifts every
                episode and takes the median over episodes; 50 draws, ``routine_null_draws``), LZ
                entropy rate S, S_unc and Fano predictability on the 30-minute symbol sequence
                concatenated over episodes with the within-agent shuffle null (``LZ_SHUFFLE_DRAWS``, and
                one exact draw where an agent's sequence holds a single symbol). A periodogram that is
                NaN on every episode stays NaN (``periodogram_*_above_band`` empty).
  shock (6)     H1c: ``routine.shock_recovery`` per evaluation seed against the unshocked control
                (``drive_gap_post``, ``reformation``, ``recovery_ticks``), median over seeds, for every
                shock pass present; stochastic pass only. The GREEDY-CLOCK reference row scores the
                same shocks from ``<root>/A-S1-N<N>-<shock>/GREEDY-CLOCK/`` (written by
                ``evaluate_grid.py --passes shocks``) against its unshocked seeds.
  swap (7)      H3c: the swap-test JSD per kind and its bootstrap CI, read from the swap pass
                (stochastic pass only).
  dashboard (8) mean drive, IQM evaluation return, policy entropy, time-budget fractions
                (``dashboard.summary``) and the degeneracy flags from ``flags.json`` when present.

Passes (``--passes``): stochastic (the primary, default), argmax (the secondary decoding; gain, cmi,
dol, social, routine, dashboard), init_narrow and init_wide (the H3d dispersion levels; dol and
dashboard), or ``all``. Rows are keyed (condition, label, seed, pass); a rerun replaces its rows and
never duplicates them, and a partial ``--sections`` rerun keeps every other section's columns from
the JSON on disk. The gain and cmi JSONs record the reference signature (the reference files and
scope that fixed the tercile and quintile edges) and are recomputed when it changes; the edges
themselves are columns (``edge_terciles_E_1`` ...), so the binning behind every gain is on record.

Baselines (RANDOM, GREEDY-STATE, GREEDY-CLOCK, GREEDY-CLOCK-WORK under ``<root>/A-S1-N<N>/``) get the
same sections over the same 32 evaluation seeds, as one pooled row each (``seed`` = "pooled"), so
H1a's paired contrast reads GREEDY-CLOCK's gain from the same reduction. ``--n-perm`` (default 200)
and ``--jobs`` (processes; keep 1 while a grid trains) control cost; ``--time`` prints per-section
seconds for the projection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hamlet.config import HOME, SOCIAL, ZONE_NAMES, HamletConfig  # noqa: E402
import analysis_constants as K  # noqa: E402
from hamlet.evaluate import BASELINES, evaluation_seeds  # noqa: E402
from hamlet.metrics import dashboard, routine, social, specialisation  # noqa: E402
from hamlet.metrics.common import DEFAULTS, after_burn_in, null_summary  # noqa: E402
from hamlet.metrics.stats import iqm  # noqa: E402

SECTIONS = {"gain": 1, "cmi": 2, "dol": 3, "social": 4, "routine": 5, "shock": 6, "swap": 7, "dashboard": 8}
# Bumped whenever the gain reduction changes, so cached gain/cmi JSONs from an older reduction are redone.
REDUCTION_VERSION = "per-evaluation-episode-primary-2026-09-05"
BINNINGS = ((3, "quantile", "terciles"), (5, "quantile", "quintiles"), (3, "fixed", "fixed"))
CLOCK = ("hour_bin",)
STATE = ("state_bin",)
CLOCK_COND = ("state_bin", "open_code", "night")
STATE_COND = ("hour_bin", "open_code", "night")
SILHOUETTE_FLOOR = 0.25
THIN = 5
TIME_SHIFT_DRAWS = 500      # the plan's draw count for the co-presence time-shift null
PASS_DIRS = {"stochastic": "stochastic", "argmax": "argmax", "init_narrow": "init_narrow", "init_wide": "init_wide"}
INIT_PASSES = ("init_narrow", "init_wide")           # H3d levels beside the stochastic pass (the default level)
INIT_SECTIONS = ("dol", "dashboard")                 # what the init-dispersion rows carry
PRIMARY_ONLY_SECTIONS = ("shock", "swap")            # sections that pair against or read the stochastic pass
SECONDARY_DRAWS = 50                                 # draws for the secondary routine nulls (regularity, periodogram, LZ shuffle)
AGENTMEDIAN_DRAWS = 50                               # draws for the candidate per-agent-in-episode gain (its value does not use them)
# The LZ shuffle null checks an identity the plan states analytically (S under the null equals S_unc), so it needs
# few draws; the match-length estimator is also at its slowest on the near-constant sequences a collapsed argmax
# policy produces (2 s a draw against 0.01 s for a varied one), which is where the cost would otherwise land.
LZ_SHUFFLE_DRAWS = 10


# ---------------------------------------------------------------------------------------------- frames
def load_episodes(files: Sequence[Path]) -> list[pd.DataFrame]:
    return [pd.read_parquet(f, engine="pyarrow") for f in files]


def pooled_per_agent(episodes: Sequence[pd.DataFrame], burn_in_days: int) -> pd.DataFrame:
    """Days 1-5 of every episode, concatenated, with ``day`` re-indexed so every agent-day is distinct across episodes."""
    parts = []
    n_days = None
    for k, df in enumerate(episodes):
        sub = after_burn_in(df, burn_in_days).copy()
        if n_days is None:
            n_days = int(df["day"].max()) + 1
        sub["day"] = sub["day"].to_numpy() + k * n_days
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def per_agent_median(values: dict[int, float]) -> float:
    arr = np.asarray([v for v in values.values() if np.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _agent_rows(df: pd.DataFrame, agent: int) -> pd.DataFrame:
    """One agent's rows (boolean indexing, so it works inside a comprehension)."""
    return df[df["agent"].to_numpy() == agent]


# ---------------------------------------------------------------------------------------------- sections
def cut_points(ref: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """The tercile and quintile edges of E, F, C in the reference frame (reported with the metrics)."""
    return {name: {col: np.quantile(ref[col].to_numpy(dtype=np.float64), np.arange(1, k) / k).tolist() for col in ("E", "F", "C")}
            for name, k in (("terciles", 3), ("quintiles", 5))}


def section_gain(episodes, ref: pd.DataFrame, cfg: HamletConfig, rng, n_perm: int,
                 agentmedian: bool = False) -> dict[str, Any]:
    """H1a and H1b under the plan's registered per-evaluation-episode reduction.

    Registered: one count table per evaluation episode, pooled over
    the eight agents, giving one value per (run, evaluation seed). The row carries the median over the
    evaluation seeds and ``per_eval_seed`` carries the values the paired contrast needs, since H1a's d_s
    is the median over evaluation seeds of the difference against GREEDY-CLOCK. This reproduces the
    reference values the plan registers: GREEDY-CLOCK 0.031 bits per tick at terciles, z 5.23.

    ``*_pooled32_*`` is the robustness column the plan names: each agent's 32 episodes pooled before
    estimation (GREEDY-CLOCK 0.058, GREEDY-CLOCK-WORK 0.103), to which the 0.02 threshold does not apply.

    ``agentmedian`` adds a diagnostic the plan does not register: the gain per agent within an episode,
    median over agents. The plan pools agents because a per-agent table is too sparse here (about 1,200
    ticks against up to 864 cells) and says the pooling is conservative when agents keep different hours;
    this column is how that claim can be checked on a trained policy. On the reference policies the
    sparsity shows plainly, GREEDY-CLOCK reading -0.004 against 0.031. Off by default: it costs about
    half again as long as the whole section and nothing in the analysis plan reads it.
    """
    out: dict[str, Any] = {"per_agent": {}, "cut_points": cut_points(ref),
                           "primary_reading": "one count table per evaluation episode, pooled over agents"}
    row: dict[str, float] = {}
    for name, edges in out["cut_points"].items():
        for col, values in edges.items():
            for i, v in enumerate(values):
                row[f"edge_{name}_{col}_{i + 1}"] = float(v)
    per_eval_seed: dict[str, list[float]] = {"eval_seed": [float(df["seed"].iloc[0]) for df in episodes]}

    # --- the registered reduction: one value per evaluation episode, median over the evaluation seeds ---
    for k, edges, name in BINNINGS:
        binned = [routine.add_bins(df, ref=ref, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night,
                                   n_state_bins=k, state_edges=edges) for df in episodes]
        clock = [routine.predictive_gain(b, "zone", CLOCK, CLOCK_COND, n_perm=n_perm, rng=rng,
                                         burn_in_days=cfg.burn_in_days) for b in binned]
        per_eval_seed[f"clock_gain_{name}_bits"] = [r["value"] for r in clock]
        per_eval_seed[f"clock_gain_{name}_z"] = [r["z"] for r in clock]
        row[f"clock_gain_{name}_bits"] = _nanmedian([r["value"] for r in clock])
        row[f"clock_gain_{name}_z"] = _nanmedian([r["z"] for r in clock])
        row[f"clock_gain_{name}_null_mean"] = _nanmedian([r["null_mean"] for r in clock])
        if name != "terciles":
            continue
        state = [routine.predictive_gain(b, "zone", STATE, STATE_COND, n_perm=n_perm, rng=rng,
                                         burn_in_days=cfg.burn_in_days) for b in binned]
        per_eval_seed["state_gain_bits"] = [r["value"] for r in state]
        per_eval_seed["state_gain_z"] = [r["z"] for r in state]
        row["state_gain_bits"] = _nanmedian([r["value"] for r in state])
        row["state_gain_z"] = _nanmedian([r["z"] for r in state])
        if not agentmedian:
            continue
        # the unregistered per-agent diagnostic, at the registered binning only; nothing in the plan reads
        # it, so its null runs at AGENTMEDIAN_DRAWS rather than the full n_perm
        med = {"clock_gain_terciles_agentmedian_bits": [], "clock_gain_terciles_agentmedian_z": [],
               "state_gain_agentmedian_bits": [], "state_gain_agentmedian_z": []}
        for b in binned:
            agents = sorted(b["agent"].unique())
            for prefix, given, cond in (("clock_gain_terciles", CLOCK, CLOCK_COND), ("state_gain", STATE, STATE_COND)):
                r = [routine.predictive_gain(_agent_rows(b, a), "zone", given, cond, n_perm=AGENTMEDIAN_DRAWS,
                                             rng=rng, burn_in_days=cfg.burn_in_days) for a in agents]
                # finite-filtered, as per_agent_median is: one agent whose null has no spread gives a
                # non-finite z, and an unfiltered median would take the whole run to NaN (H1b counts on this z)
                med[f"{prefix}_agentmedian_bits"].append(_nanmedian([x["value"] for x in r]))
                med[f"{prefix}_agentmedian_z"].append(_nanmedian([x["z"] for x in r]))
        per_eval_seed.update(med)
        row.update({col: _nanmedian(values) for col, values in med.items()})

    # --- robustness: each agent's evaluation episodes pooled before estimation (the plan's other reading) ---
    pooled = pooled_per_agent(episodes, cfg.burn_in_days)
    for k, edges, name in BINNINGS:
        binned = routine.add_bins(pooled, ref=ref, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night,
                                  n_state_bins=k, state_edges=edges)
        vals, zs = {}, {}
        for agent in sorted(binned["agent"].unique()):
            r = routine.predictive_gain(_agent_rows(binned, agent), "zone", CLOCK, CLOCK_COND,
                                        n_perm=n_perm, rng=rng, burn_in_days=0)
            vals[int(agent)], zs[int(agent)] = r["value"], r["z"]
        out["per_agent"][f"clock_gain_{name}_pooled32"] = vals
        row[f"clock_gain_{name}_pooled32_bits"] = per_agent_median(vals)
        row[f"clock_gain_{name}_pooled32_z"] = per_agent_median(zs)
        if name != "terciles":
            continue
        svals, szs = {}, {}
        for agent in sorted(binned["agent"].unique()):
            r = routine.predictive_gain(_agent_rows(binned, agent), "zone", STATE, STATE_COND,
                                        n_perm=n_perm, rng=rng, burn_in_days=0)
            svals[int(agent)], szs[int(agent)] = r["value"], r["z"]
        out["per_agent"]["state_gain_pooled32"] = svals
        row["state_gain_pooled32_bits"] = per_agent_median(svals)
        row["state_gain_pooled32_z"] = per_agent_median(szs)

    out["per_eval_seed"] = per_eval_seed
    out["row"] = row
    return out


def section_cmi(episodes, ref: pd.DataFrame, cfg: HamletConfig, rng, n_perm: int) -> dict[str, Any]:
    pooled = pooled_per_agent(episodes, cfg.burn_in_days)
    binned = routine.add_bins(pooled, ref=ref, job_open=cfg.job_open, market_open=cfg.market_open, night=cfg.night)
    row, per = {}, {}
    for name, given, cond, null in (("clockmi_relabel", CLOCK, CLOCK_COND, "relabel"), ("clockmi_block", CLOCK, CLOCK_COND, "block"),
                                    ("clockmi_tick", CLOCK, CLOCK_COND, "tick"), ("statemi_block", STATE, STATE_COND, "block")):
        vals, zs = {}, {}
        for agent in sorted(binned["agent"].unique()):
            r = routine.conditional_mi(binned[binned["agent"] == agent], "zone", given, cond, n_perm=n_perm, rng=rng, burn_in_days=0, null=null)
            vals[int(agent)], zs[int(agent)] = r["value"], r["z"]
        per[name] = vals
        row[f"{name}_bits"] = per_agent_median(vals)
        row[f"{name}_z"] = per_agent_median(zs)
    return {"row": row, "per_agent": per}


def cluster_with_floor(counts: np.ndarray) -> tuple[np.ndarray, float]:
    labels, sil = specialisation.cluster_agents(counts)
    if not np.isfinite(sil) or sil < SILHOUETTE_FLOOR:
        return np.zeros(counts.shape[0], dtype=np.int64), float(sil)
    return np.asarray(labels), float(sil)


def section_dol(episodes, cfg: HamletConfig, rng, n_perm: int, replay: Optional[dict[str, np.ndarray]]) -> dict[str, Any]:
    per_ep = []
    labels_all, ks = [], []
    pooled_counts = None
    for df in episodes:
        seed = int(df["seed"].iloc[0])
        per_day = specialisation.time_budget(df, burn_in_days=cfg.burn_in_days, n_agents=cfg.N, per_day=True)
        counts = per_day.sum(axis=1)
        pooled_counts = counts if pooled_counts is None else pooled_counts + counts
        day = specialisation.dol_indiv(per_day, n_perm=n_perm, rng=rng, null="home_shuffle")
        multi = specialisation.dol_indiv(per_day, n_perm=n_perm, rng=rng, null="multinomial")
        rec = {"eval_seed": seed, "dol_indiv": day["value"], "dol_z_dayshuffle": day["z"], "dol_null_mean_dayshuffle": day["null_mean"],
               "dol_z_multinomial": multi["z"], "specialisation_index": specialisation.specialisation_index(counts)}
        key = f"seed{seed}"
        if replay is not None and key in replay:
            th = specialisation.dol_indiv(per_day, n_perm=n_perm, rng=rng, null="trait_and_home_shuffle", replay_counts=list(replay[key]))
            rec["dol_z_trait_home_shuffle"] = th["z"]
        labels, sil = cluster_with_floor(counts)
        rec["n_clusters"], rec["silhouette"] = int(len(set(labels.tolist()))), sil
        labels_all.append(labels)
        ks.append(rec["n_clusters"])
        per_ep.append(rec)
    table = pd.DataFrame(per_ep)
    m = len(labels_all)
    pair_vals = []
    for i in range(m):
        for j in range(i + 1, m):
            pair_vals.append(0.0 if (ks[i] == 1 or ks[j] == 1) else specialisation.ari(labels_all[i], labels_all[j]))
    ci = specialisation.ari_across_episodes_ci(labels_all, n_boot=DEFAULTS.n_boot, rng=rng) if m >= 2 else {}
    row = {
        "dol_indiv": float(table["dol_indiv"].median()), "dol_z_dayshuffle": float(table["dol_z_dayshuffle"].median()),
        "dol_z_multinomial": float(table["dol_z_multinomial"].median()), "specialisation_index": float(table["specialisation_index"].median()),
        "dol_pooled32": specialisation.dol_indiv_value(pooled_counts) if pooled_counts is not None else float("nan"),
        # the registered statistic and its CI use the library's definition (a pair of one-cluster labellings scores 1);
        # ari_across_episodes_k1zero counts such pairs as chance (0) and is reported beside it
        "ari_across_episodes": specialisation.ari_across_episodes(labels_all) if m >= 2 else float("nan"),
        "ari_across_episodes_k1zero": float(np.mean(pair_vals)) if pair_vals else float("nan"),
        "ari_ci_low": ci.get("ci_low", float("nan")), "ari_ci_high": ci.get("ci_high", float("nan")),
        "n_clusters_median": float(table["n_clusters"].median()), "silhouette_median": float(table["silhouette"].median()),
        "episodes_k1": int((table["n_clusters"] == 1).sum()),
    }
    if "dol_z_trait_home_shuffle" in table:
        row["dol_z_trait_home_shuffle"] = float(table["dol_z_trait_home_shuffle"].median())
        row["n_trait_home_shuffle_episodes"] = int(table["dol_z_trait_home_shuffle"].notna().sum())
    return {"row": row, "per_episode": table.to_dict("records"), "labels": [l.tolist() for l in labels_all]}


def section_social(episodes, cfg: HamletConfig, rng, n_perm: int) -> dict[str, Any]:
    per_ep, curves = [], []
    for df in episodes:
        counts = social.copresence_counts(df, SOCIAL, cfg.N, cfg.burn_in_days)
        null = social.time_shift_null(df, SOCIAL, cfg.N, TIME_SHIFT_DRAWS, rng, cfg.burn_in_days)
        oe = np.asarray(social.observed_over_expected(df, SOCIAL, cfg.N, cfg.burn_in_days), dtype=np.float64)
        iu = np.triu_indices(cfg.N, k=1)
        per_ep.append({
            "eval_seed": int(df["seed"].iloc[0]),
            "copresence_density": social.thresholded_density(counts, null),
            "mean_oe": float(np.nanmean(oe[iu])) if np.isfinite(oe[iu]).any() else float("nan"),
            "arrival_phase_variance": float(social.arrival_phase_variance(df, SOCIAL, cfg.ticks_per_day, cfg.burn_in_days)),
            "contention_per_agent_day": float(social.contention_per_agent_day(df, cfg.burn_in_days)),
        })
        curves.append(np.asarray(social.gossip_curve(df), dtype=np.float64))
    table = pd.DataFrame(per_ep)
    length = min(len(c) for c in curves) if curves else 0
    curve = np.mean([c[:length] for c in curves], axis=0) if length else np.array([])
    row = {k: float(table[k].median()) for k in ("copresence_density", "mean_oe", "arrival_phase_variance", "contention_per_agent_day")}
    row.update({f"gossip_day{d}": float(curve[d]) for d in range(len(curve))})
    return {"row": row, "per_episode": table.to_dict("records"), "gossip_curve": curve.tolist()}


def circular_shift_days(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each agent-day's zone sequence rotated by its own random offset (the per-day circular-shift null)."""
    out = df.sort_values(["agent", "day", "t_day"]).reset_index(drop=True)
    zone = out["zone"].to_numpy().copy()
    starts = np.flatnonzero(np.r_[True, (out["agent"].to_numpy()[1:] != out["agent"].to_numpy()[:-1]) | (out["day"].to_numpy()[1:] != out["day"].to_numpy()[:-1])])
    ends = np.r_[starts[1:], len(out)]
    for a, b in zip(starts, ends):
        zone[a:b] = np.roll(zone[a:b], int(rng.integers(0, b - a)))
    out["zone"] = zone
    return out


def _agent_sequence(df: pd.DataFrame, agent: int, burn_in_days: int) -> np.ndarray:
    data = after_burn_in(df, burn_in_days)
    sub = data[data["agent"].to_numpy() == agent].sort_values("t")
    return sub["zone"].to_numpy()


def _nanmedian(values) -> float:
    """Median over the finite entries; NaN (silently) when there are none."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def section_routine(episodes, cfg: HamletConfig, rng, n_perm: int, draws: int = SECONDARY_DRAWS) -> dict[str, Any]:
    """Secondary routine metrics; the null is the null of the per-seed statistic: every draw shifts every episode
    (per-day circular shift) and takes the median over episodes, ``draws`` times."""
    T = cfg.ticks_per_day
    zones = (("SOCIAL", SOCIAL), ("HOME", HOME))
    reg_vals = [routine.regularity(df, T, slot=1, burn_in_days=cfg.burn_in_days) for df in episodes]
    per_vals = {name: [routine.periodogram_power_at_day(df, zone, T, cfg.burn_in_days) for df in episodes] for name, zone in zones}
    reg_null, per_null = [], {name: [] for name, _ in zones}
    for _ in range(draws):
        shifted = [circular_shift_days(df, rng) for df in episodes]
        reg_null.append(_nanmedian([routine.regularity(sh, T, slot=1, burn_in_days=cfg.burn_in_days) for sh in shifted]))
        for name, zone in zones:
            per_null[name].append(_nanmedian([routine.periodogram_power_at_day(sh, zone, T, cfg.burn_in_days) for sh in shifted]))
    reg = _nanmedian(reg_vals)
    row = {"regularity": reg, "regularity_null_mean": float(np.nanmean(reg_null)),
           "regularity_z": null_summary(reg, np.asarray(reg_null))["z"] if np.isfinite(reg) else float("nan"),
           "routine_null_draws": int(draws)}
    for name, _ in zones:
        vals, null = np.asarray(per_vals[name], dtype=np.float64), np.asarray(per_null[name], dtype=np.float64)
        med = _nanmedian(vals)
        finite_null = null[np.isfinite(null)]
        lo, hi = (np.percentile(finite_null, [2.5, 97.5]) if finite_null.size else (np.nan, np.nan))
        row[f"periodogram_{name}"] = med
        row[f"periodogram_{name}_band_low"], row[f"periodogram_{name}_band_high"] = float(lo), float(hi)
        row[f"periodogram_{name}_above_band"] = (bool(med > hi) if np.isfinite(med) and np.isfinite(hi) else None)
        row[f"periodogram_{name}_finite_episodes"] = int(np.isfinite(vals).sum())
    # LZ entropy rate on the 30-minute symbol sequence, concatenated over episodes per agent
    s_vals, unc_vals, fano_vals, null_vals = [], [], [], []
    for agent in range(cfg.N):
        seq = np.concatenate([_agent_sequence(df, agent, cfg.burn_in_days)[::THIN] for df in episodes])
        S = routine.lz_entropy_rate(seq)
        S_unc = routine.uncorrelated_entropy(seq)
        n_sym = int(len(np.unique(seq)))
        s_vals.append(S); unc_vals.append(S_unc)
        fano_vals.append(routine.fano_predictability(S, n_sym) if n_sym > 1 else float("nan"))
        # a one-symbol sequence is its own every permutation, so one draw is the exact null, not an estimate;
        # it also matters for cost, because the match-length estimator is slowest on a constant sequence
        lz_draws = 1 if n_sym < 2 else LZ_SHUFFLE_DRAWS
        null_vals.append(np.mean([routine.lz_entropy_rate(rng.permutation(seq)) for _ in range(lz_draws)]))
    row.update({"lz_entropy_rate": float(np.median(s_vals)), "entropy_uncorrelated": float(np.median(unc_vals)),
                "lz_over_unc": float(np.median(np.asarray(s_vals) / np.maximum(np.asarray(unc_vals), 1e-9))),
                "fano_predictability": float(np.nanmedian(fano_vals)), "lz_shuffle_null": float(np.median(null_vals)),
                "lz_shuffle_draws": int(LZ_SHUFFLE_DRAWS), "lz_degenerate_agents": int(sum(1 for v in unc_vals if v <= 0.0))})
    return {"row": row, "per_agent": {"lz": s_vals, "unc": unc_vals, "fano": fano_vals}}


def shock_pairs_of_run(run_dir: Path) -> dict[str, tuple[Sequence[Path], str]]:
    """``{shock: (shocked files, scope)}`` from the run's eval manifest, directories derived from ``run_dir``."""
    path = run_dir / "eval" / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"passes": {}}
    out = {}
    for key, entry in manifest.get("passes", {}).items():
        if key.startswith("shock_") and entry.get("status") == "complete":
            files = sorted((run_dir / "eval" / key).glob("*/*/seed*_ep0.parquet"))
            out[entry["shock"]] = (files, entry.get("scope", ""))
    return out


def section_shock(shocked: dict[str, tuple[Sequence[Path], str]], control_files: Sequence[Path], cfg: HamletConfig) -> dict[str, Any]:
    controls = {int(pd.read_parquet(f, columns=["seed"]).iloc[0, 0]): f for f in control_files}
    row, per = {}, {}
    for shock, (files, scope) in shocked.items():
        recs = []
        for f in files:
            shocked = pd.read_parquet(f, engine="pyarrow")
            seed = int(shocked["seed"].iloc[0])
            if seed not in controls:
                continue
            control = pd.read_parquet(controls[seed], engine="pyarrow")
            r = routine.shock_recovery(shocked, control, shock_day=cfg.shock_day, ticks_per_day=cfg.ticks_per_day, burn_in_days=cfg.burn_in_days)
            recs.append({"eval_seed": seed, **{k: (float(v) if v is not None else np.nan) for k, v in r.items()}})
        if not recs:
            print(f"warning: shock {shock}: no shocked/control pairs found", flush=True)
            continue
        table = pd.DataFrame(recs)
        per[shock] = table.to_dict("records")
        for col in ("drive_gap_post", "drive_gap_shock", "reformation", "jaccard_post", "jaccard_ref_control"):
            row[f"{shock}_{col}"] = float(table[col].median())
        row[f"{shock}_recovery_ticks_median"] = float(table["recovery_ticks"].median(skipna=True))
        row[f"{shock}_recovered_fraction"] = float(table["recovery_ticks"].notna().mean())
        row[f"{shock}_n_pairs"] = int(len(table))
        row[f"{shock}_scope"] = scope
    return {"row": row, "per_episode": per}


def section_swap(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "eval" / "swap" / "swap.json"
    if not path.exists():
        return {"row": {}}
    data = json.loads(path.read_text())
    row = {}
    for kind, entry in data["kinds"].items():
        row[f"swap_jsd_{kind}"] = entry["mean_jsd"]
        row[f"swap_jsd_{kind}_ci_low"], row[f"swap_jsd_{kind}_ci_high"] = entry["ci_low"], entry["ci_high"]
    row["swap_n_observations"] = data["n_observations"]
    return {"row": row}


def section_dashboard(episodes, files: Sequence[Path], cfg: HamletConfig, flags_path: Optional[Path]) -> dict[str, Any]:
    rows = [dashboard.summary(df, cfg.burn_in_days, traits_file=dashboard.traits_file_for(f)).iloc[0] for df, f in zip(episodes, files)]
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include="number")
    row = {f"{k}": float(v) for k, v in numeric.median().items() if k not in ("seed", "episode", "n_agents")}
    row["return_iqm"] = iqm(np.asarray([dashboard.mean_return(df) for df in episodes]))
    if flags_path and flags_path.exists():
        flags = json.loads(flags_path.read_text())
        row.update({f"flag_{k}": bool(flags.get(k)) for k in ("few_zones", "single_zone", "low_entropy", "below_greedy", "below_greedy_iqm", "degenerate")})
    return {"row": row}


# ---------------------------------------------------------------------------------------------- units
def reference_frame(files: Sequence[Path]) -> pd.DataFrame:
    """The pooled evaluation data whose quantiles define the state bins (columns E, F, C only)."""
    return pd.concat([pd.read_parquet(f, columns=["E", "F", "C"], engine="pyarrow") for f in files], ignore_index=True)


def compute_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """One (run or baseline, pass): every requested section, resumable per section; returns the per-seed row."""
    files = [Path(f) for f in unit["files"]]
    metrics_dir = Path(unit["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg = HamletConfig(**{k: v for k, v in unit["config"].items() if k in HamletConfig.__dataclass_fields__})
    for name in ("job_open", "market_open", "night"):
        if isinstance(getattr(cfg, name), list):
            setattr(cfg, name, tuple(getattr(cfg, name)))
    seed_key = unit["seed_key"]
    row: dict[str, Any] = {"condition": unit["condition"], "label": unit["label"], "seed": unit["seed"], "pass": unit["pass"],
                           "n_episodes": len(files)}
    # every section already on disk contributes its columns, so a partial --sections rerun never blanks the
    # others; cached_keys remembers which columns came from which section, so recomputing one can drop the
    # columns its previous run wrote and no longer emits
    cached_keys: dict[str, set[str]] = {}
    for section in SECTIONS:
        cached = metrics_dir / f"{section}.json"
        if not cached.exists():
            continue
        try:
            payload = json.loads(cached.read_text())
        except json.JSONDecodeError:
            continue
        if section in ("gain", "cmi") and payload.get("ref_signature", {}).get("reduction") != REDUCTION_VERSION:
            # an older reduction's numbers would otherwise be carried forward under the current column names;
            # leaving them out makes the gap visible instead, and a rerun with that section fills it
            print(f"warning: {cached} predates reduction {REDUCTION_VERSION}; its columns are left blank until "
                  f"--sections {section} is rerun", flush=True)
            continue
        cached_keys[section] = set(payload.get("row", {}))
        row.update(payload.get("row", {}))
    timings: dict[str, float] = {}
    episodes = None
    ref = None
    ref_signature = {"ref_scope": unit.get("ref_scope"), "ref_files": sorted(unit["ref_files"]), "n_perm": unit["n_perm"],
                     "reduction": REDUCTION_VERSION, "agentmedian": bool(unit.get("agentmedian"))}
    for section in unit["sections"]:
        if unit["pass"] != "stochastic" and section in PRIMARY_ONLY_SECTIONS:
            continue                                   # shocks pair against, and swap reads, the stochastic pass only
        out_path = metrics_dir / f"{section}.json"
        if out_path.exists() and not unit["force"]:
            cached = json.loads(out_path.read_text())
            same_ref = section not in ("gain", "cmi") or cached.get("ref_signature") == ref_signature
            if same_ref:
                row.update(cached["row"])
                continue
        t0 = time.perf_counter()
        rng = np.random.default_rng([seed_key, SECTIONS[section]])
        if episodes is None:
            episodes = load_episodes(files)
        if section in ("gain", "cmi") and ref is None:
            ref = reference_frame([Path(f) for f in unit["ref_files"]])
        if section == "gain":
            result = section_gain(episodes, ref, cfg, rng, unit["n_perm"], agentmedian=bool(unit.get("agentmedian")))
            result["ref_signature"] = ref_signature
        elif section == "cmi":
            result = section_cmi(episodes, ref, cfg, rng, unit["n_perm"])
            result["ref_signature"] = ref_signature
        elif section == "dol":
            replay = None
            rp = Path(unit["replay_path"]) if unit.get("replay_path") else None
            if rp and rp.exists():
                with np.load(rp) as z:
                    replay = {k: z[k] for k in z.files}
            result = section_dol(episodes, cfg, rng, unit["n_perm"], replay)
        elif section == "social":
            result = section_social(episodes, cfg, rng, unit["n_perm"])
        elif section == "routine":
            result = section_routine(episodes, cfg, rng, unit["n_perm"])
        elif section == "shock":
            if unit.get("run_dir"):
                result = section_shock(shock_pairs_of_run(Path(unit["run_dir"])), files, cfg)
            elif unit.get("shock_dirs"):
                shocked = {shock: (sorted(Path(d).glob("seed*_ep0.parquet")), "reference") for shock, d in unit["shock_dirs"].items()}
                result = section_shock(shocked, files, cfg)
            else:
                result = {"row": {}}
        elif section == "swap":
            result = section_swap(Path(unit["run_dir"])) if unit.get("run_dir") else {"row": {}}
        else:
            result = section_dashboard(episodes, files, cfg, Path(unit["run_dir"]) / "flags.json" if unit.get("run_dir") else None)
        result["seconds"] = time.perf_counter() - t0
        result["n_perm"] = unit["n_perm"]
        out_path.write_text(json.dumps(result, indent=1, default=float))
        timings[section] = result["seconds"]
        for stale in cached_keys.pop(section, set()) - set(result["row"]):
            row.pop(stale, None)                 # a column this section used to write and no longer does
        row.update(result["row"])
    row["_timings"] = timings
    return row


KEY = ("condition", "label", "seed", "pass")


def merge_per_seed(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace every row of ``old`` belonging to a unit ``new`` recomputes.

    The key is the unit, never the row, so a rerun over fewer evaluation seeds cannot leave the surplus
    rows of the previous run behind. Keys are compared as strings because ``seed`` mixes ints and "pooled".
    """
    if new.empty:
        return old

    def key_of(df: pd.DataFrame) -> pd.Series:
        return df[list(KEY)].astype(str).agg("|".join, axis=1)
    keep = old[~key_of(old).isin(set(key_of(new)))]
    return pd.concat([keep, new], ignore_index=True)


def per_eval_seed_table(units: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per (run, evaluation seed) from the cached gain JSONs: the input of the H1a and H1b contrasts.

    The plan pairs by evaluation seed: per training seed s, d_s is the median over the 32
    evaluation seeds e of gain_A(s, e) - gain_GC(e), and the bootstrap CI and the Wilcoxon run on the eight
    d_s. That needs the gain at every (run, evaluation seed), which ``per_seed.csv`` has already reduced away.
    """
    rows = []
    for unit in units:
        path = Path(unit["metrics_dir"]) / "gain.json"
        if not path.exists():
            continue
        try:
            per = json.loads(path.read_text()).get("per_eval_seed") or {}
        except json.JSONDecodeError:
            continue
        for i, eval_seed in enumerate(per.get("eval_seed") or []):
            row = {"condition": unit["condition"], "label": unit["label"], "seed": unit["seed"],
                   "pass": unit["pass"], "eval_seed": int(eval_seed)}
            row.update({col: values[i] for col, values in per.items() if col != "eval_seed" and i < len(values)})
            rows.append(row)
    return pd.DataFrame(rows)


def build_units(root: Path, conditions: Optional[Sequence[str]], seeds: Optional[Sequence[int]], passes: Sequence[str],
                sections: Sequence[str], n_eval: int, n_perm: int, force: bool, ref_scope: str, baselines: bool,
                agentmedian: bool = False) -> list[dict[str, Any]]:
    eval_seeds = set(evaluation_seeds(n_eval))
    units = []
    run_dirs = sorted(p.parent for p in root.glob("*/seed*/selection.json"))
    if conditions:
        run_dirs = [d for d in run_dirs if d.parent.name in set(conditions)]
    if seeds is not None:
        run_dirs = [d for d in run_dirs if int(d.name.removeprefix("seed")) in set(seeds)]
    by_condition: dict[str, list[Path]] = {}
    for d in run_dirs:
        by_condition.setdefault(d.parent.name, []).append(d)
    for cond, dirs in by_condition.items():
        for pass_name in passes:
            files_by_run = {}
            for d in dirs:
                files = sorted(f for f in (d / "eval" / PASS_DIRS[pass_name]).glob("*/*/seed*_ep0.parquet")
                               if int(f.name[4:].split("_")[0]) in eval_seeds)
                if files:
                    files_by_run[d] = files
            cond_ref = [f for fs in files_by_run.values() for f in fs]
            unit_sections = [s for s in sections if pass_name not in INIT_PASSES or s in INIT_SECTIONS]
            for d, files in files_by_run.items():
                meta = json.loads((d / "metadata.json").read_text())
                seed = int(d.name.removeprefix("seed"))
                units.append({
                    "condition": cond, "label": files[0].parent.name, "seed": seed, "seed_key": seed, "pass": pass_name,
                    "files": [str(f) for f in files], "ref_files": [str(f) for f in (cond_ref if ref_scope == "condition" else files)],
                    "ref_scope": ref_scope, "metrics_dir": str(d / "metrics" / pass_name), "run_dir": str(d),
                    "config": meta["hamlet_config"], "sections": unit_sections, "n_perm": n_perm, "force": force,
                    "agentmedian": agentmedian,
                    "replay_path": str(d / "eval" / "swap" / "trait_home_shuffle.npz"),
                })
    # The baseline folder is named for the population of this root, not of today's config.
    a_s1 = f"A-S1-N{K.root_n_agents(root)}"
    if baselines and (conditions is None or a_s1 in conditions):
        for policy in BASELINES:
            files = sorted(f for f in (root / a_s1 / policy).glob("seed*_ep0.parquet") if int(f.name[4:].split("_")[0]) in eval_seeds)
            if not files:
                continue
            shock_dirs = {shock: str(root / f"{a_s1}-{shock}" / policy) for shock in ("jobs_closed_d4", "energy_x2_d4")
                          if (root / f"{a_s1}-{shock}" / policy).exists()}
            units.append({
                "condition": a_s1, "label": policy, "seed": "pooled", "seed_key": 10_000, "pass": "stochastic",
                "files": [str(f) for f in files], "ref_files": [str(f) for f in files], "ref_scope": "policy",
                "metrics_dir": str(root / "metrics" / "baselines" / policy), "run_dir": None, "config": json.loads(HamletConfig().to_json()),
                "sections": [s for s in sections if s != "swap"], "n_perm": n_perm, "force": force, "replay_path": None,
                "agentmedian": agentmedian,
                "shock_dirs": shock_dirs,
            })
    return units


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--sections", nargs="+", default=["all"], choices=["all", *SECTIONS])
    parser.add_argument("--passes", nargs="+", default=["stochastic"], choices=[*PASS_DIRS, "all"],
                        help="stochastic (primary), argmax, init_narrow, init_wide (H3d levels; dol and dashboard sections only), or all")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-eval-seeds", type=int, default=32)
    parser.add_argument("--n-perm", type=int, default=DEFAULTS.n_perm)
    parser.add_argument("--jobs", type=int, default=1, help="processes; keep 1 while a grid trains")
    parser.add_argument("--ref-scope", choices=["condition", "run"], default="run",
                        help="pooled data for the state-bin edges; the plan fixes them on the run's own "
                             "32 evaluation episodes, so 'run' is the registered default")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--agentmedian", action="store_true",
                        help="add the unregistered per-agent-within-episode gain diagnostic (the plan pools agents; "
                             "this is how its conservativeness claim can be checked). Off by default: it costs about "
                             "half again as long as the gain section and no analysis-plan table reads it")
    parser.add_argument("--time", action="store_true", help="print per-section seconds")
    args = parser.parse_args(argv)

    root = Path(args.root)
    sections = list(SECTIONS) if "all" in args.sections else list(args.sections)
    passes = list(PASS_DIRS) if "all" in args.passes else args.passes
    units = build_units(root, args.conditions, args.seeds, passes, sections, args.n_eval_seeds, args.n_perm, args.force,
                        args.ref_scope, not args.no_baselines, args.agentmedian)
    if not units:
        print(f"nothing to compute under {root}: run scripts/evaluate_grid.py first")
        return
    print(f"{len(units)} unit(s) x {len(sections)} section(s), n_perm {args.n_perm}, {args.jobs} process(es)")
    t0 = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(compute_unit, units))
    else:
        rows = [compute_unit(u) for u in units]
    timings = [r.pop("_timings") for r in rows]
    table = pd.DataFrame(rows)
    out_dir = root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "per_seed.csv"
    if out.exists():
        table = merge_per_seed(pd.read_csv(out), table)
    table.to_csv(out, index=False)
    print(f"{len(rows)} row(s) merged into {out} ({len(table)} rows total) in {time.perf_counter() - t0:.1f} s")
    eval_table = per_eval_seed_table(units) if "gain" in sections else pd.DataFrame()
    if eval_table.empty:
        if "gain" in sections:
            print("no gain.json was produced by this invocation, so metrics/per_eval_seed.csv is left as it stands")
    else:
        eval_out = out_dir / "per_eval_seed.csv"
        if eval_out.exists() and eval_out.stat().st_size:
            # merged on the unit key, not on (unit, evaluation seed): recomputing a unit replaces every one of
            # its evaluation-seed rows, so a rerun over fewer seeds cannot leave the surplus behind
            eval_table = merge_per_seed(pd.read_csv(eval_out), eval_table)
        eval_table.to_csv(eval_out, index=False)
        print(f"{len(eval_table)} (run, evaluation seed) row(s) written to {eval_out}: the H1a and H1b paired differences")
    if args.time:
        for u, t in zip(units, timings):
            print(f"  {u['condition']} {u['label']} seed {u['seed']}: " + ", ".join(f"{k} {v:.1f}s" for k, v in t.items()) + f" | total {sum(t.values()):.1f}s")


if __name__ == "__main__":
    main()
