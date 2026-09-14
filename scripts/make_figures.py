"""Every figure and results table of the paper, from the tables the post-grid chain writes.

    uv run python scripts/make_figures.py                  # runs/ -> runs/figures/, runs/reports/
    uv run python scripts/make_figures.py --root runs --only h1a learning

Matplotlib only, one style for every panel, SVG and PNG side by side. Captions go to
runs/figures/captions.md and say "exploratory" wherever docs/analysis_plan.md section 8.1 says the
result is exploratory: argmax decoding, learning curves, A-noclock contrasts, the simple contrasts
of the 2x2, every secondary metric and all of H4.

A figure whose inputs are missing, because a pass did not run or a condition was never trained, is
skipped with a line saying which input was absent, and its caption records that. Nothing crashes on
missing data: an untested hypothesis is a result (section 12).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hamlet.config import HamletConfig  # noqa: E402

# Condition names carry the population size and follow the run root; set_population() rebinds
# them once --root is known, so an archived root at another N is read under its own names.
NN = HamletConfig().n_agents
A_S1 = f"A-S1-N{NN}"


def set_population(root) -> None:
    global NN, A_S1
    NN = K.root_n_agents(root)
    A_S1 = f"A-S1-N{NN}"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analysis_constants as K  # noqa: E402
from hamlet.config import ZONE_NAMES, HamletConfig  # noqa: E402
from hamlet.metrics.stats import bootstrap_ci, iqm  # noqa: E402

BASELINES = ("RANDOM", "GREEDY-STATE", "GREEDY-CLOCK", "GREEDY-CLOCK-WORK")
ZONE_COLOURS = {"HOME": "#4C6EF5", "FARM": "#2F9E44", "OFFICE": "#F08C00", "CANTEEN": "#E8590C",
                "SOCIAL": "#AE3EC9", "MARKET": "#1098AD", "TRANSIT": "#ADB5BD"}
EXPLORATORY = "Exploratory: not a confirmatory test (section 8.1)."

STYLE = {"figure.dpi": 120, "savefig.dpi": 160, "font.size": 9, "axes.grid": True,
         "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
         "axes.spines.right": False, "legend.frameon": False, "figure.constrained_layout.use": True}

captions: list[str] = []
skipped: list[str] = []


def save(fig, out_dir: Path, name: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    captions.append(f"**{name}** — {caption}")
    print(f"  wrote {out_dir / name}.svg and .png")


def skip(name: str, why: str) -> None:
    skipped.append(f"**{name}** — not produced: {why}")
    captions.append(f"**{name}** — not produced: {why}")
    print(f"  skipped {name}: {why}")


def runs_of(t: pd.DataFrame, condition: str, pass_name: str = "stochastic") -> pd.DataFrame:
    if t.empty:
        return t
    return t[(t["condition"] == condition) & (t["pass"] == pass_name) & (~t["label"].isin(BASELINES))]


def iqm_ci(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(v[0]), np.nan, np.nan
    low, high, point = bootstrap_ci(v, stat=iqm, n=int(K.N_BOOT.value))
    return float(point), float(low), float(high)


# ------------------------------------------------------------------ 1. actograms
def fig_actograms(root: Path, out: Path) -> None:
    cfg = HamletConfig()
    picks = []
    for cond, label in ((A_S1, f"{A_S1} (learned)"), (None, "GREEDY-CLOCK (reference)")):
        if cond:
            f = sorted((root / cond).glob("seed*/eval/stochastic/*/*/seed10000_ep0.parquet"))
        else:
            f = sorted((root / A_S1 / "GREEDY-CLOCK").glob("seed10000_ep0.parquet"))
        if f:
            picks.append((label, f[0]))
    if len(picks) < 1:
        return skip("01_actograms", "no stochastic evaluation Parquet for either policy")
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 3.4), squeeze=False)
    order = [ZONE_NAMES.index(z) for z in ZONE_NAMES]
    cmap = matplotlib.colors.ListedColormap([ZONE_COLOURS[z] for z in ZONE_NAMES])
    for ax, (label, path) in zip(axes[0], picks):
        df = pd.read_parquet(path, columns=["agent", "day", "t_day", "zone"])
        d = df[(df["agent"] == 0) & (df["day"] >= cfg.burn_in_days)]
        grid = d.pivot_table(index="day", columns="t_day", values="zone", aggfunc="first").to_numpy()
        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(ZONE_NAMES) - 1, interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("tick of day"); ax.set_ylabel("day"); ax.grid(False)
    handles = [matplotlib.patches.Patch(color=ZONE_COLOURS[z], label=K.display(z)) for z in ZONE_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=len(ZONE_NAMES), fontsize=7)
    save(fig, out, "01_actograms",
         "Occupancy of one agent over days 1-5 on evaluation seed 10000, learned policy beside the "
         f"reference scheduler. {EXPLORATORY}")


# ------------------------------------------------------------------ 2. time budgets
def fig_time_budget(per_seed: pd.DataFrame, out: Path) -> None:
    cols = [f"mean_frac_{z}" for z in ZONE_NAMES if f"mean_frac_{z}" in per_seed.columns]
    if not cols:
        cols = [f"frac_{z}" for z in ZONE_NAMES if f"frac_{z}" in per_seed.columns]
    if per_seed.empty or not cols:
        return skip("02_time_budget", "per_seed.csv carries no time-budget columns")
    groups: list[tuple[str, pd.DataFrame]] = []
    for cond in sorted(per_seed["condition"].unique()):
        r = runs_of(per_seed, cond)
        if not r.empty:
            groups.append((cond, r))
    for b in BASELINES:
        r = per_seed[(per_seed["label"] == b) & (per_seed["pass"] == "stochastic")]
        if not r.empty:
            groups.append((b, r))
    if not groups:
        return skip("02_time_budget", "no runs or references in per_seed.csv")
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(groups), 3.6))
    bottoms = np.zeros(len(groups))
    for c in cols:
        zone = c.replace("mean_frac_", "").replace("frac_", "")
        vals = np.array([iqm_ci(g[c].to_numpy())[0] for _, g in groups], dtype=np.float64)
        vals = np.nan_to_num(vals)
        ax.bar(range(len(groups)), vals, bottom=bottoms, label=K.display(zone),
               color=ZONE_COLOURS.get(zone, "#888"))
        bottoms += vals
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("share of agent-ticks (IQM over seeds)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, out, "02_time_budget",
         "Time budget over the seven occupancy symbols, IQM over training seeds, days 1-5. "
         "Conditions first, then the reference policies.")


# ------------------------------------------------------------------ 3. H1a
def fig_h1a(per_eval: pd.DataFrame, out: Path) -> None:
    binnings = [("terciles", "terciles (registered)"), ("quintiles", "quintiles"), ("fixed", "fixed edges")]
    have = [(b, t) for b, t in binnings if f"clock_gain_{b}_bits" in per_eval.columns] if not per_eval.empty else []
    if not have:
        return skip("03_h1a", "per_eval_seed.csv carries no clock-gain columns")
    gc_all = per_eval[per_eval["label"] == "GREEDY-CLOCK"]
    a_all = runs_of(per_eval, A_S1)
    if gc_all.empty or a_all.empty:
        return skip("03_h1a", f"GREEDY-CLOCK reference or {A_S1} runs absent from per_eval_seed.csv")
    thr = float(K.H1A_THRESHOLD.value)
    fig, axes = plt.subplots(1, len(have), figsize=(3.5 * len(have), 3.4), squeeze=False, sharey=True)
    for ax, (b, title) in zip(axes[0], have):
        col = f"clock_gain_{b}_bits"
        ref = gc_all.set_index("eval_seed")[col]
        d, labels = [], []
        for seed, g in a_all.groupby("seed"):
            s = g.set_index("eval_seed")[col]
            shared = s.index.intersection(ref.index)
            if len(shared):
                d.append(float(np.median(s.loc[shared] - ref.loc[shared]))); labels.append(str(seed))
        if not d:
            ax.set_title(f"{title}\n(no shared seeds)"); continue
        d = np.asarray(d)
        ax.scatter(range(len(d)), d, s=28, color="#1c7ed6", zorder=3, label="per training seed")
        m, lo, hi = iqm_ci(d)
        ax.errorbar([len(d) + 0.6], [m], yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None,
                    fmt="o", color="#212529", capsize=4, zorder=4, label="IQM with CI")
        ax.axhline(thr, color="#e03131", ls="--", lw=1.2, label=f"threshold {thr}")
        ax.axhline(0.0, color="#868e96", lw=0.8, label="GREEDY-CLOCK's zero")
        ax.set_xticks(list(range(len(d))) + [len(d) + 0.6])
        ax.set_xticklabels(labels + ["IQM"], fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0][0].set_ylabel("d_s, bits per tick")
    axes[0][-1].legend(fontsize=7, loc="best")
    save(fig, out, "03_h1a",
         f"H1a: per-seed d_s, the median over evaluation seeds of the clock gain of A minus "
         f"GREEDY-CLOCK's, at three binnings. The dashed line is the registered threshold "
         f"{thr} bits per tick (sections {K.H1A_THRESHOLD.section}); the thin line is GREEDY-CLOCK's "
         "own gain, the zero of H1 by definition. Terciles is the registered binning; the other two "
         "panels are the pre-declared robustness variants.")


# ------------------------------------------------------------------ 4. H1b and H2c
def fig_state_gain(per_seed: pd.DataFrame, out: Path) -> None:
    if per_seed.empty or "state_gain_bits" not in per_seed.columns:
        return skip("04_state_gain", "per_seed.csv has no state_gain_bits")
    groups = []
    for cond in sorted(per_seed["condition"].unique()):
        r = runs_of(per_seed, cond)
        if not r.empty:
            groups.append((cond, r))
    for b in BASELINES:
        r = per_seed[(per_seed["label"] == b) & (per_seed["pass"] == "stochastic")]
        if not r.empty:
            groups.append((b, r))
    if not groups:
        return skip("04_state_gain", "no rows in per_seed.csv")
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(groups), 3.4))
    for i, (name, g) in enumerate(groups):
        v = g["state_gain_bits"].to_numpy(dtype=np.float64)
        m, lo, hi = iqm_ci(v)
        ax.scatter([i] * len(v), v, s=18, color="#adb5bd", zorder=2)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None,
                    fmt="o", color="#212529", capsize=4, zorder=3)
        if "state_gain_z" in g.columns:
            z = float(np.nanmedian(g["state_gain_z"].to_numpy(dtype=np.float64)))
            ax.annotate(f"z {z:.1f}", (i, m), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7)
    ax.axhline(0.0, color="#868e96", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("state gain, bits per tick")
    save(fig, out, "04_state_gain",
         "H1b and H2c: held-out predictive gain of the state bins beyond the clock, per condition "
         "and reference, with the median relabel z annotated. Points are training seeds; the bar is "
         "the IQM with its bootstrap CI.")


# ------------------------------------------------------------------ 5. H1c
def fig_shock(root: Path, per_seed: pd.DataFrame, out: Path) -> None:
    cfg = HamletConfig()
    shocked = sorted(root.glob(f"{A_S1}/seed*/eval/shock_jobs_closed_d4/*/*/seed*_ep0.parquet"))
    control = sorted(root.glob(f"{A_S1}/seed*/eval/stochastic/*/*/seed*_ep0.parquet"))
    gc = sorted((root / A_S1 / "GREEDY-CLOCK").glob("seed*_ep0.parquet"))
    if not shocked:
        return skip("05_h1c_shock", "the jobs_closed_d4 pass did not run")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for label, files, colour, ls in ((f"{A_S1} shocked", shocked, "#e03131", "-"),
                                     (f"{A_S1} control", control, "#1c7ed6", "-"),
                                     ("GREEDY-CLOCK", gc, "#868e96", "--")):
        if not files:
            continue
        d = pd.concat([pd.read_parquet(f, columns=["day", "D"]) for f in files[:32]], ignore_index=True)
        d = d[d["day"] >= cfg.burn_in_days].groupby("day")["D"].mean()
        ax.plot(d.index, d.to_numpy(), marker="o", ms=3.5, color=colour, ls=ls, label=label)
    ax.axvline(cfg.shock_day, color="#f08c00", lw=1.0, ls=":", label="shock day")
    ax.set_xlabel("day"); ax.set_ylabel("mean drive"); ax.legend(fontsize=7)
    save(fig, out, "05_h1c_shock",
         "H1c: mean drive by day, the shocked run against its paired control on the same evaluation "
         "seeds, with the reference scheduler for scale. The registered statistic is the day-5 gap, "
         f"whose IQM CI upper bound must fall below {K.H1C_BOUND.value}.")


# ------------------------------------------------------------------ 6. H3
def fig_h3(per_seed: pd.DataFrame, out: Path) -> None:
    panels: list[tuple[str, str, str]] = []
    if "dol_z_dayshuffle" in per_seed.columns:
        panels.append(("dol_z_dayshuffle", "H3a: within-episode DOL z", f"z bar {K.Z_BAR.value}"))
    if "ari_across_episodes" in per_seed.columns:
        panels.append(("ari_across_episodes", "H3b: cross-episode ARI", "zero"))
    swap = [c for c in per_seed.columns if c.startswith("swap_jsd_")]
    if swap:
        panels.append((swap[0], "H3c: swap-test JSD", "zero"))
    if per_seed.empty or not panels:
        return skip("06_h3", "per_seed.csv carries no specialisation columns")
    conds = [c for c in sorted(per_seed["condition"].unique()) if not runs_of(per_seed, c).empty]
    if not conds:
        return skip("06_h3", "no trained runs in per_seed.csv")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), squeeze=False)
    for ax, (col, title, bar) in zip(axes[0], panels):
        for i, cond in enumerate(conds):
            v = runs_of(per_seed, cond)[col].to_numpy(dtype=np.float64)
            m, lo, hi = iqm_ci(v)
            ax.scatter([i] * len(v), v, s=18, color="#adb5bd", zorder=2)
            ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None,
                        fmt="o", color="#212529", capsize=4, zorder=3)
        ax.axhline(float(K.Z_BAR.value) if "z" in col else 0.0, color="#e03131", ls="--", lw=1.0)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
    save(fig, out, "06_h3",
         "H3: specialisation. Within-episode division of labour against the agent-day shuffle null, "
         "cross-episode ARI of the time-budget cluster labels, and the counterfactual swap test. "
         "Points are training seeds; bars are IQM with bootstrap CI; the dashed line is the "
         f"registered bar (z {K.Z_BAR.value}, or zero).")


def fig_h3d(per_seed: pd.DataFrame, out: Path) -> None:
    levels = [("init_narrow", "narrow"), ("stochastic", "default"), ("init_wide", "wide")]
    have = [(p, n) for p, n in levels if not per_seed[per_seed["pass"] == p].empty] if not per_seed.empty else []
    if len(have) < 3 or "dol_z_dayshuffle" not in per_seed.columns:
        return skip("06b_h3d_dispersion", "the init-dispersion passes did not run")
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for cond in [c for c in sorted(per_seed["condition"].unique()) if not runs_of(per_seed, c).empty]:
        for seed in sorted(runs_of(per_seed, cond)["seed"].unique()):
            ys = []
            for p, _ in levels:
                sub = per_seed[(per_seed["condition"] == cond) & (per_seed["seed"] == seed) & (per_seed["pass"] == p)]
                ys.append(float(sub["dol_z_dayshuffle"].iloc[0]) if len(sub) else np.nan)
            ax.plot([0, 1, 2], ys, marker="o", ms=3.5, alpha=0.8, label=f"{cond} seed {seed}")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([n for _, n in levels])
    ax.set_xlabel("initial-state dispersion"); ax.set_ylabel("within-episode DOL z")
    ax.legend(fontsize=6, ncol=2)
    save(fig, out, "06b_h3d_dispersion",
         "H3d: within-episode division-of-labour z against the initial-state dispersion level, one "
         "line per training seed. The registered statistic is the per-seed Spearman correlation.")


# ------------------------------------------------------------------ 7. learning curves
def fig_learning(root: Path, out: Path) -> None:
    runs = sorted(p.parent for p in root.glob("*/seed*/progress.csv") if "_gate2" not in str(p))
    if not runs:
        return skip("07_learning_curves", "no progress.csv under the run root")
    by_cond: dict[str, list[pd.DataFrame]] = {}
    for r in runs:
        try:
            d = pd.read_csv(r / "progress.csv")
        except Exception:  # noqa: BLE001
            continue
        by_cond.setdefault(r.parent.name, []).append(d)
    if not by_cond:
        return skip("07_learning_curves", "no readable progress.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(by_cond), 2)))
    for (cond, frames), colour in zip(sorted(by_cond.items()), colours):
        for ax, col, ylab in ((axes[0], "episode_return_iqm", "training return (IQM)"),
                              (axes[1], "entropy", "policy entropy on decision ticks, nats")):
            series = []
            for d in frames:
                if col not in d.columns:
                    continue
                s = d[["update", col]].dropna().set_index("update")[col]
                series.append(s)
            if not series:
                continue
            wide = pd.concat(series, axis=1)
            med = wide.median(axis=1)
            ax.plot(med.index, med.to_numpy(), color=colour, lw=1.3, label=cond)
            if wide.shape[1] > 1:
                ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color=colour, alpha=0.15)
            ax.set_xlabel("update"); ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7)
    save(fig, out, "07_learning_curves",
         "Training return and decision-tick policy entropy per checkpoint, median over training "
         "seeds with the seed range shaded, one colour per condition. The plateau criterion reads "
         f"the last five finite checkpoint returns against a {K.PLATEAU_TOLERANCE.value:.0%} "
         f"tolerance (section {K.PLATEAU_TOLERANCE.section}). {EXPLORATORY}")


# ------------------------------------------------------------------ 8. Park comparator
def fig_park(per_seed: pd.DataFrame, out: Path) -> None:
    cols = [c for c in ("copresence_density", "mean_copresence_oe") if c in per_seed.columns]
    if per_seed.empty or not cols:
        return skip("08_park_comparator", "per_seed.csv carries no co-presence columns")
    groups = [(c, runs_of(per_seed, c)) for c in sorted(per_seed["condition"].unique())]
    groups = [(n, g) for n, g in groups if not g.empty]
    for b in BASELINES:
        g = per_seed[(per_seed["label"] == b) & (per_seed["pass"] == "stochastic")]
        if not g.empty:
            groups.append((b, g))
    if not groups:
        return skip("08_park_comparator", "no rows to compare")
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(groups), 3.4))
    col = cols[0]
    for i, (name, g) in enumerate(groups):
        v = g[col].to_numpy(dtype=np.float64)
        m, lo, hi = iqm_ci(v)
        ax.scatter([i] * len(v), v, s=18, color="#adb5bd", zorder=2)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None,
                    fmt="o", color="#212529", capsize=4, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(col.replace("copresence", "co-presence").replace("_", " "))
    save(fig, out, "08_park_comparator",
         f"Co-presence density at {K.display('SOCIAL')} beside the reference policies, for "
         "comparison with Park et "
         "al. (2023). **Edges here are co-presence, not mutual knowledge**: two agents share an edge "
         "when they are active in the same zone at the same tick, which is a weaker relation than "
         f"the acquaintance graph Park et al. report. {EXPLORATORY}")


# ------------------------------------------------------------------ tables
def h3_overall(reports: Path) -> str:
    """The one-line reading of the H3 family, from the confirmatory table."""
    csv = reports / "confirmatory.csv"
    if not csv.exists():
        return ""
    t = pd.read_csv(csv)
    t = t[t["scope"] == "all runs"].set_index("test")
    def word(name: str) -> str:
        return str(t.loc[name, "decision"]) if name in t.index else "not run"
    def size(name: str) -> str:
        if name not in t.index or pd.isna(t.loc[name, "iqm"]):
            return ""
        return f" ({t.loc[name, 'iqm']:.3f}, CI {t.loc[name, 'ci_low']:.3f} to {t.loc[name, 'ci_high']:.3f})"
    return (
        "## H3 overall: identity without roles (section 7)\n\n"
        f"- **H3a** within-episode division of labour: **{word('H3a')}**{size('H3a')}\n"
        f"- **H3b(i)** cross-episode ARI under S1: **{word('H3b(i)')}**{size('H3b(i)')}\n"
        f"- **H3b(ii)** ARI, S1 minus S0: **{word('H3b(ii)')}**{size('H3b(ii)')}\n"
        f"- **H3c** counterfactual swap test: **{word('H3c')}**{size('H3c')}\n"
        f"- **H3d** dispersion trend: **{word('H3d')}**{size('H3d')}\n\n"
        "Agents are identifiable across episodes and the swap test confirms the identity is used, "
        "but within an episode they do not divide the zones between them, and the division does not "
        "strengthen with initial-state dispersion. Identity without roles.\n"
    )


def write_tables(root: Path, per_seed: pd.DataFrame, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    flags = sorted(root.glob("*/seed*/flags.json"))
    rows = []
    for f in flags:
        if "_gate2" in str(f):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({"condition": f.parent.parent.name, "seed": f.parent.name.removeprefix("seed"),
                     **{k: d.get(k) for k in ("few_zones", "single_zone", "low_entropy", "below_greedy",
                                              "below_greedy_iqm", "degenerate", "n_episodes")}})
    ft = reports / "flags_table.md"
    if rows:
        t = pd.DataFrame(rows).sort_values(["condition", "seed"])
        ft.write_text("# Degeneracy flags\n\nSection 10: flagged runs are reported, never dropped; "
                      "every confirmatory table is shown with and without them.\n\n"
                      f"{t.to_markdown(index=False)}\n\n{int(t['degenerate'].sum())} of {len(t)} run(s) flagged.\n")
        print(f"  wrote {ft}")
    else:
        ft.write_text("# Degeneracy flags\n\nNo flags.json found; run `report_grid.py --section flags`.\n")

    man = root / "manifest.csv"
    et = reports / "energy.md"
    if man.exists():
        m = pd.read_csv(man)
        e = m.dropna(subset=["energy_kwh"]) if "energy_kwh" in m.columns else m.iloc[0:0]
        total_kwh = float(e["energy_kwh"].sum()) if len(e) else float("nan")
        total_kg = float(e["emissions_kg"].sum()) if "emissions_kg" in e.columns and len(e) else float("nan")
        hours = float(m["wall_clock_min"].fillna(0).sum()) / 60 if "wall_clock_min" in m.columns else float("nan")
        et.write_text(
            "# Energy and carbon\n\nFrom each run's own codecarbon record, as collected in "
            "`runs/manifest.csv` (process-level tracking, GPU excluded).\n\n"
            f"| quantity | value |\n|---|---|\n| runs with an energy record | {len(e)} of {len(m)} |\n"
            f"| total energy | {total_kwh:.4f} kWh |\n| total emissions | {total_kg * 1000:.1f} g CO2e |\n"
            f"| total wall-clock | {hours:.2f} h |\n")
        print(f"  wrote {et}")
    else:
        et.write_text("# Energy and carbon\n\nNo runs/manifest.csv; run `aggregate_manifest.py`.\n")

    conf = reports / "confirmatory.md"
    parts = ["# Results tables, in the paper's order\n", h3_overall(reports), ""]
    parts.append(conf.read_text() if conf.exists() else
                 "## Confirmatory contrasts\n\nNot produced; run `scripts/confirmatory.py`.\n")
    parts.append("\n\n" + ft.read_text())
    parts.append("\n\n" + et.read_text())
    (reports / "results_tables.md").write_text("\n".join(parts))
    print(f"  wrote {reports / 'results_tables.md'}")


# ------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="run root to read and write under. No default, deliberately: a default of\n                             'runs' let a half-analysed rerun be read into a finished paper.")
    parser.add_argument("--out", default=None, help="figure directory (default <root>/figures)")
    parser.add_argument("--only", nargs="*", default=None, help="subset: actograms budget h1a state shock h3 learning park")
    args = parser.parse_args(argv)
    root = Path(args.root)
    set_population(root)
    out = Path(args.out) if args.out else root / "figures"
    reports = root / "reports"

    ps_path, pe_path = root / "metrics" / "per_seed.csv", root / "metrics" / "per_eval_seed.csv"
    per_seed = pd.read_csv(ps_path) if ps_path.exists() else pd.DataFrame()
    per_eval = pd.read_csv(pe_path) if pe_path.exists() else pd.DataFrame()

    want = set(args.only) if args.only else None
    plt.rcParams.update(STYLE)
    print(f"figures -> {out}")
    if want is None or "actograms" in want:
        fig_actograms(root, out)
    if want is None or "budget" in want:
        fig_time_budget(per_seed, out)
    if want is None or "h1a" in want:
        fig_h1a(per_eval, out)
    if want is None or "state" in want:
        fig_state_gain(per_seed, out)
    if want is None or "shock" in want:
        fig_shock(root, per_seed, out)
    if want is None or "h3" in want:
        fig_h3(per_seed, out); fig_h3d(per_seed, out)
    if want is None or "learning" in want:
        fig_learning(root, out)
    if want is None or "park" in want:
        fig_park(per_seed, out)

    out.mkdir(parents=True, exist_ok=True)
    (out / "captions.md").write_text(
        "# Figure captions\n\nEvery caption marked exploratory is a result section 8.1 places "
        "outside the confirmatory families.\n\n"
        + h3_overall(root / "reports").replace("## H3 overall", "### H3 overall") + "\n"
        + "\n\n".join(captions) + "\n")
    print(f"  wrote {out / 'captions.md'}")
    write_tables(root, per_seed, reports)
    if skipped:
        print(f"\n{len(skipped)} figure(s) skipped for missing inputs:")
        for s in skipped:
            print("  " + s.split(" — ", 1)[1])


if __name__ == "__main__":
    main()
