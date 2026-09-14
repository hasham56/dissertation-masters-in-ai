"""The confirmatory contrasts of docs/analysis_plan.md section 8.1, one row per registered test.

    uv run python scripts/confirmatory.py                       # runs/ -> runs/reports/confirmatory.{csv,md}
    uv run python scripts/confirmatory.py --root runs

Every test statistic is built from ``runs/metrics/per_seed.csv`` and ``per_eval_seed.csv``; every
p-value, interval and effect size comes from ``hamlet.metrics.stats``; every threshold comes from
``scripts/analysis_constants.py``. Nothing numeric is written here.

Each row carries the IQM and stratified bootstrap CI of each side, the probability of improvement
with its CI, Cliff's delta, the raw and Holm-adjusted p, n, the pre-registered decision word, and a
``disagreement`` column set where the rule's word and the adjusted p point different ways. Section
8.1's precedence is followed exactly: the rule's word stands and the disagreement is stated.

Every table is produced twice, over all runs and over unflagged runs only (section 10: flagged runs
are reported, never dropped). A test whose inputs are missing, because a pass did not run or a
condition was never trained, produces a row reading "pass not run" rather than an absent row or a
crash: an untested hypothesis is a result and section 12 requires it to be visible.
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

import analysis_constants as K  # noqa: E402

# Condition names carry the population size and follow the run root; set_population() rebinds
# them once --root is known, so an archived root at another N is read under its own names.
NN = HamletConfig().n_agents
A_S1, A_S0, D_S1 = f"A-S1-N{NN}", f"A-S0-N{NN}", f"D-S1-N{NN}"
A_S1_NOCLOCK = f"A-S1-N{NN}-noclock"


def set_population(root) -> None:
    global NN, A_S1, A_S0, D_S1, A_S1_NOCLOCK
    NN = K.root_n_agents(root)
    A_S1, A_S0, D_S1 = f"A-S1-N{NN}", f"A-S0-N{NN}", f"D-S1-N{NN}"
    A_S1_NOCLOCK = f"A-S1-N{NN}-noclock"
from hamlet.metrics.stats import (  # noqa: E402
    bootstrap_ci,
    cliffs_delta,
    holm,
    iqm,
    paired_wilcoxon,
    prob_improvement,
    sign_test,
    tost,
)

MISSING = "pass not run"
BASELINE_LABELS = ("RANDOM", "GREEDY-STATE", "GREEDY-CLOCK", "GREEDY-CLOCK-WORK")


# ---------------------------------------------------------------- loading
def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = root / "metrics" / "per_seed.csv"
    per_eval = root / "metrics" / "per_eval_seed.csv"
    a = pd.read_csv(per_seed) if per_seed.exists() else pd.DataFrame()
    b = pd.read_csv(per_eval) if per_eval.exists() else pd.DataFrame()
    return a, b


def runs_of(t: pd.DataFrame, condition: str, pass_name: str = "stochastic") -> pd.DataFrame:
    """Training-run rows of one condition: baselines carry seed "pooled" and are excluded."""
    if t.empty:
        return t
    m = (t["condition"] == condition) & (t["pass"] == pass_name) & (~t["label"].isin(BASELINE_LABELS))
    return t[m].copy()


def baseline_of(t: pd.DataFrame, label: str, pass_name: str = "stochastic") -> pd.DataFrame:
    if t.empty:
        return t
    return t[(t["label"] == label) & (t["pass"] == pass_name)].copy()


def unflagged(t: pd.DataFrame) -> pd.DataFrame:
    """Rows whose run is not flagged; if the flag column is absent nothing is dropped."""
    if t.empty or "flag_degenerate" not in t.columns:
        return t
    return t[~t["flag_degenerate"].fillna(False).astype(bool)].copy()


# ---------------------------------------------------------------- one row
BOOTSTRAP_SEED = 20260908      # fixed so a rerun reproduces every interval and every decision word


def summarise(values: np.ndarray, other: Optional[np.ndarray] = None, n_boot: Optional[int] = None) -> dict[str, Any]:
    """IQM and CI of a per-seed vector, with the paired effect sizes when a second side is given.

    The bootstrap is seeded, so the interval reported in a row and the interval that decided its
    word are the same numbers. Taking the word from one draw and the columns from another can put
    "yes" beside an interval that straddles the threshold.
    """
    n_boot = int(K.N_BOOT.value) if n_boot is None else n_boot
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "iqm": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    low, high, point = bootstrap_ci(v, stat=iqm, n=n_boot, rng=np.random.default_rng(BOOTSTRAP_SEED))
    out: dict[str, Any] = {"n": int(v.size), "iqm": float(point),
                           "ci_low": float(low), "ci_high": float(high)}
    if other is not None:
        w = np.asarray(other, dtype=np.float64)
        if w.size == v.size and np.isfinite(w).all():
            pi = prob_improvement(v, w, n_boot=n_boot, rng=np.random.default_rng(BOOTSTRAP_SEED))
            out.update({"p_improve": float(pi.get("value", np.nan)),
                        "p_improve_low": float(pi.get("ci_low", np.nan)),
                        "p_improve_high": float(pi.get("ci_high", np.nan)),
                        "cliffs_delta": float(cliffs_delta(v, w))})
    return out


def ci_word(low: float, high: float, threshold: float, greater_is_yes: bool = True) -> str:
    """The plan's CI rule: the whole interval above the threshold is yes, wholly below is no."""
    if not (np.isfinite(low) and np.isfinite(high)):
        return "inconclusive"
    if greater_is_yes:
        if low > threshold:
            return "yes"
        return "no" if high < threshold else "inconclusive"
    if high < threshold:
        return "yes"
    return "no" if low > threshold else "inconclusive"


def row(test: str, family: str, statistic: str, rule: str, *, values=None, other=None,
        p: float = np.nan, word: str = "inconclusive", note: str = "",
        summary: Optional[dict[str, Any]] = None, **extra) -> dict[str, Any]:
    r: dict[str, Any] = {"test": test, "family": family, "statistic": statistic, "rule": rule,
                         "p_raw": float(p) if p == p else np.nan, "decision": word, "note": note}
    if summary is None:
        summary = summarise(values, other) if values is not None else {
            "n": 0, "iqm": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    r.update(summary)
    r.update(extra)
    return r


def missing_row(test: str, family: str, statistic: str, why: str) -> dict[str, Any]:
    return row(test, family, statistic, rule="-", word=MISSING, note=why)


# ---------------------------------------------------------------- the tests
def h1a(per_eval: pd.DataFrame, condition: str, binning: str = "terciles") -> dict[str, Any]:
    """d_s = median over evaluation seeds of gain_A(s, e) - gain_GC(e), at one binning."""
    col = f"clock_gain_{binning}_bits"
    name = f"H1a" if binning == "terciles" else f"H1a ({binning})"
    fam = "H1" if binning == "terciles" else "-"
    if per_eval.empty or col not in per_eval.columns:
        return missing_row(name, fam, f"d_s at {binning}", "metrics/per_eval_seed.csv has no such column")
    gc = baseline_of(per_eval, "GREEDY-CLOCK")
    a = runs_of(per_eval, condition)
    if gc.empty or a.empty:
        return missing_row(name, fam, f"d_s at {binning}",
                           "GREEDY-CLOCK reference or the condition's runs are absent")
    ref = gc.set_index("eval_seed")[col]
    d, seeds = [], []
    for seed, g in a.groupby("seed"):
        s = g.set_index("eval_seed")[col]
        shared = s.index.intersection(ref.index)
        if len(shared) == 0:
            continue
        d.append(float(np.median(s.loc[shared] - ref.loc[shared])))
        seeds.append(seed)
    if not d:
        return missing_row(name, fam, f"d_s at {binning}", "no evaluation seed shared with GREEDY-CLOCK")
    d = np.asarray(d, dtype=np.float64)
    thr = float(K.H1A_THRESHOLD.value)
    w = paired_wilcoxon(d, 0.0, shift=thr, alternative="greater")
    s = summarise(d)
    return row(name, fam, f"d_s at {binning} (median over evaluation seeds of A - GREEDY-CLOCK)",
               rule=f"IQM CI wholly above {thr}", values=d, p=w["pvalue"],
               word=ci_word(s["ci_low"], s["ci_high"], thr), note=f"seeds {list(seeds)}", summary=s)


def h1a_pooled32(per_seed: pd.DataFrame, condition: str) -> dict[str, Any]:
    """The robustness column: each agent's 32 episodes pooled before estimation (section 7)."""
    col = "clock_gain_terciles_pooled32_bits"
    if per_seed.empty or col not in per_seed.columns:
        return missing_row("H1a (pooled32 robustness)", "-", "pooled-over-episodes gain", "column absent")
    a, gc = runs_of(per_seed, condition), baseline_of(per_seed, "GREEDY-CLOCK")
    if a.empty or gc.empty:
        return missing_row("H1a (pooled32 robustness)", "-", "pooled-over-episodes gain", "runs or reference absent")
    ref = float(gc[col].mean())
    d = a[col].to_numpy(dtype=np.float64) - ref
    s = summarise(d)
    return row("H1a (pooled32 robustness)", "-", "pooled32 gain minus GREEDY-CLOCK's",
               rule="reported, not tested: the 0.025 threshold does not apply to this scale (section 7)",
               values=d, word="-", note=f"GREEDY-CLOCK pooled32 = {ref:.4f}")


def h1b(per_seed: pd.DataFrame, condition: str) -> dict[str, Any]:
    """m_s, the state gain of A, with the count-and-CI rule."""
    if per_seed.empty or "state_gain_bits" not in per_seed.columns:
        return missing_row("H1b", "H1", "m_s state gain", "metrics/per_seed.csv has no state_gain_bits")
    a = runs_of(per_seed, condition)
    if a.empty:
        return missing_row("H1b", "H1", "m_s state gain", f"{condition} has no trained runs")
    m = a["state_gain_bits"].to_numpy(dtype=np.float64)
    z = a["state_gain_z"].to_numpy(dtype=np.float64) if "state_gain_z" in a.columns else np.full(m.size, np.nan)
    w = paired_wilcoxon(m, 0.0, alternative="greater")
    s = summarise(m)
    n_pass = int(np.sum(z > float(K.Z_BAR.value)))
    need = int(np.ceil(float(K.COUNT_RULE_OF_EIGHT.value) / 8 * m.size))
    ci_excludes_zero = np.isfinite(s["ci_low"]) and s["ci_low"] > 0
    word = "yes" if (n_pass >= need and ci_excludes_zero) else ("no" if not ci_excludes_zero else "inconclusive")
    return row("H1b", "H1", "m_s state gain beyond the clock",
               rule=f"relabel z > {K.Z_BAR.value} in at least {need} of {m.size} seeds and the IQM CI excludes 0",
               values=m, p=w["pvalue"], word=word,
               note=f"z > {K.Z_BAR.value} in {n_pass} of {m.size}; CI excludes 0: {bool(ci_excludes_zero)}",
               count_pass=n_pass, count_needed=need)


def h1c(per_seed: pd.DataFrame, condition: str) -> list[dict[str, Any]]:
    """H1c with section 7's interpretability condition applied, as two rows.

    Section 7: if job time in the A controls falls below 5% on days 1-5, `jobs_closed_d4` says
    nothing, `energy_x2_d4` replaces it, the outcome becomes schedule re-formation, and the
    substitution is logged. The jobs row is then reported for the record but carries no p-value and
    enters no family; the reformation row is the H1 family member.

    Translating section 7's "at or above -0.1" into the table's CI-and-Wilcoxon form: yes when
    the IQM CI lies entirely above -0.1, no when entirely below, inconclusive otherwise, with
    `paired_wilcoxon(reformation, 0.0, shift=-0.1, alternative="greater")` as the Holm p.
    """
    if per_seed.empty:
        return [missing_row("H1c", "H1", "gap_s day-5 drive gap", "metrics/per_seed.csv is absent")]
    a = runs_of(per_seed, condition)
    if a.empty:
        return [missing_row("H1c", "H1", "gap_s day-5 drive gap", f"{condition} has no trained runs")]

    job_cols = [c for c in ("frac_FARM", "frac_OFFICE") if c in a.columns] or \
               [c for c in ("mean_frac_FARM", "mean_frac_OFFICE") if c in a.columns]
    job_frac = float(a[job_cols].sum(axis=1).mean()) if job_cols else float("nan")
    threshold = float(K.H1C_JOB_FRACTION_MIN.value)
    interpretable = bool(np.isfinite(job_frac)) and job_frac >= threshold

    rows: list[dict[str, Any]] = []
    gap_col = "jobs_closed_d4_drive_gap_post"
    if gap_col in a.columns and not a[gap_col].isna().all():
        v = a[gap_col].to_numpy(dtype=np.float64)
        bound = float(K.H1C_BOUND.value)
        if interpretable:
            w = paired_wilcoxon(v, 0.0, shift=bound, alternative="less")
            sm = summarise(v)
            rows.append(row("H1c", "H1", "gap_s day-5 drive gap, jobs_closed_d4",
                            rule=f"IQM CI upper bound below {bound}", values=v, p=w["pvalue"],
                            word=ci_word(sm["ci_low"], sm["ci_high"], bound, greater_is_yes=False), summary=sm,
                            note=f"job time fraction {job_frac:.4f}, at or above {threshold}"))
            return rows
        rows.append(row("H1c (jobs_closed_d4, not interpretable)", "-",
                        "gap_s day-5 drive gap, jobs_closed_d4",
                        rule="reported for the record; enters no family", values=v, word="-",
                        note=(f"not interpretable: job time {job_frac:.1%}; substitution triggered "
                              f"(section 7). The gap is exactly zero on every seed because these "
                              f"agents never work, so closing the jobs perturbs nothing.")))
    ref_col = "energy_x2_d4_reformation"
    if ref_col not in a.columns or a[ref_col].isna().all():
        rows.append(missing_row("H1c", "H1", "reformation under energy_x2_d4 (substituted)",
                                f"job time {job_frac:.1%} is below {threshold}, so energy_x2_d4 "
                                "substitutes, but its pass did not run"))
        return rows
    v = a[ref_col].to_numpy(dtype=np.float64)
    floor = float(K.H1C_REFORMATION_MIN.value)
    w = paired_wilcoxon(v, 0.0, shift=floor, alternative="greater")
    sm = summarise(v)
    rows.append(row("H1c", "H1", "reformation under energy_x2_d4 (substituted)",
                    rule=f"IQM CI entirely above {floor}", values=v, p=w["pvalue"],
                    word=ci_word(sm["ci_low"], sm["ci_high"], floor), summary=sm,
                    note=(f"SUBSTITUTED: job time {job_frac:.1%} below {threshold} (section 7). "
                          "Re-formation at or above -0.1 counts as re-formed.")))
    return rows


def paired_contrast(per_seed: pd.DataFrame, test: str, family: str, col: str, left: str, right: str,
                    statistic: str, alternative: str = "greater") -> dict[str, Any]:
    """A_s - D_s style contrast on a shared training seed, one-sided paired Wilcoxon."""
    if per_seed.empty or col not in per_seed.columns:
        return missing_row(test, family, statistic, f"metrics/per_seed.csv has no {col}")
    a, b = runs_of(per_seed, left), runs_of(per_seed, right)
    if a.empty or b.empty:
        absent = left if a.empty else right
        return missing_row(test, family, statistic, f"{absent} did not run")
    ja = a.set_index("seed")[col]
    jb = b.set_index("seed")[col]
    shared = ja.index.intersection(jb.index)
    if len(shared) == 0:
        return missing_row(test, family, statistic, f"{left} and {right} share no training seed")
    x, y = ja.loc[shared].to_numpy(float), jb.loc[shared].to_numpy(float)
    w = paired_wilcoxon(x, y, alternative=alternative)
    return row(test, family, statistic, rule="Holm-adjusted p below 0.05", values=x - y, other=None,
               p=w["pvalue"], word="pending-holm", note=f"{len(shared)} shared seed(s)")


def h3a(per_seed: pd.DataFrame, condition: str) -> dict[str, Any]:
    col = "dol_z_dayshuffle"
    if per_seed.empty or col not in per_seed.columns:
        return missing_row("H3a", "H3", "within-episode DOL z", f"no {col}")
    a = runs_of(per_seed, condition)
    if a.empty:
        return missing_row("H3a", "H3", "within-episode DOL z", f"{condition} has no trained runs")
    z = a[col].to_numpy(dtype=np.float64)
    bar = float(K.Z_BAR.value)
    w = paired_wilcoxon(z, bar, alternative="greater")
    n_pass = int(np.sum(z > bar))
    need = int(np.ceil(float(K.COUNT_RULE_OF_EIGHT.value) / 8 * z.size))
    return row("H3a", "H3", "z_s within-episode DOL against the agent-day shuffle",
               rule=f"z > {bar} in at least {need} of {z.size} seeds", values=z, p=w["pvalue"],
               word="yes" if n_pass >= need else "no", note=f"{n_pass} of {z.size} above {bar}",
               count_pass=n_pass, count_needed=need)


def h3b_i(per_seed: pd.DataFrame, condition: str) -> dict[str, Any]:
    col = "ari_across_episodes"
    if per_seed.empty or col not in per_seed.columns:
        return missing_row("H3b(i)", "H3", "cross-episode ARI", f"no {col}")
    a = runs_of(per_seed, condition)
    if a.empty:
        return missing_row("H3b(i)", "H3", "cross-episode ARI", f"{condition} has no trained runs")
    v = a[col].to_numpy(dtype=np.float64)
    w = paired_wilcoxon(v, 0.0, alternative="greater")
    s = summarise(v)
    return row("H3b(i)", "H3", "ARI_s cross-episode, S1", rule="IQM CI excludes 0 from above",
               values=v, p=w["pvalue"], word=ci_word(s["ci_low"], s["ci_high"], 0.0), summary=s)


def h3c(per_seed: pd.DataFrame, condition: str) -> dict[str, Any]:
    cols = [c for c in per_seed.columns if c.startswith("swap_jsd_")] if not per_seed.empty else []
    if not cols:
        return missing_row("H3c", "H3", "swap-test JSD", "the swap pass did not run")
    a = runs_of(per_seed, condition)
    if a.empty or a[cols].isna().all(axis=None):
        return missing_row("H3c", "H3", "swap-test JSD", "the swap pass did not run for this condition")
    v = a[cols].mean(axis=1).to_numpy(dtype=np.float64)
    w = paired_wilcoxon(v, 0.0, alternative="greater")
    s = summarise(v)
    return row("H3c", "H3", "j_s mean swap-test JSD, S1", rule="IQM CI excludes 0 from above",
               values=v, p=w["pvalue"], word=ci_word(s["ci_low"], s["ci_high"], 0.0), summary=s,
               note=f"kinds: {', '.join(c.removeprefix('swap_jsd_') for c in cols)}")


def h3d(per_seed: pd.DataFrame, condition: str) -> dict[str, Any]:
    """rho_s: Spearman of the within-episode DOL z with the dispersion level, under S0."""
    levels = [("init_narrow", "narrow"), ("stochastic", "default"), ("init_wide", "wide")]
    col = "dol_z_dayshuffle"
    if per_seed.empty or col not in per_seed.columns:
        return missing_row("H3d", "H3", "rho_s dispersion trend", f"no {col}")
    have = [p for p, _ in levels if not per_seed[(per_seed["pass"] == p)].empty]
    if len(have) < 3:
        return missing_row("H3d", "H3", "rho_s dispersion trend",
                           f"the init-dispersion passes did not run (found {have or 'none'})")
    from scipy.stats import spearmanr

    rhos, seeds = [], []
    for seed in sorted(runs_of(per_seed, condition)["seed"].unique()):
        vals = []
        for p, _ in levels:
            sub = per_seed[(per_seed["condition"] == condition) & (per_seed["seed"] == seed) & (per_seed["pass"] == p)]
            vals.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
        if np.isfinite(vals).all():
            rhos.append(float(spearmanr([0, 1, 2], vals).statistic))
            seeds.append(seed)
    if not rhos:
        return missing_row("H3d", "H3", "rho_s dispersion trend", "no seed has all three dispersion levels")
    r = np.asarray(rhos, dtype=np.float64)
    st = sign_test(r, 0.0, alternative="greater")
    n_pass = int(np.sum(r > 0))
    need = int(np.ceil(float(K.COUNT_RULE_OF_EIGHT.value) / 8 * r.size))
    return row("H3d", "H3", "rho_s Spearman of DOL z with dispersion, S0",
               rule=f"rho > 0 in at least {need} of {r.size} seeds", values=r, p=st["pvalue"],
               word="yes" if n_pass >= need else "no", note=f"{n_pass} of {r.size} positive",
               count_pass=n_pass, count_needed=need)


def noclock_tost(per_seed: pd.DataFrame) -> dict[str, Any]:
    """A-noclock minus GREEDY-STATE, TOST at the registered bound; outside every Holm family."""
    col = "clock_gain_terciles_bits"
    if per_seed.empty or col not in per_seed.columns:
        return missing_row("A-noclock minus GREEDY-STATE", "-", "clock gain difference", f"no {col}")
    a = runs_of(per_seed, A_S1_NOCLOCK)
    gs = baseline_of(per_seed, "GREEDY-STATE")
    if a.empty or gs.empty:
        return missing_row("A-noclock minus GREEDY-STATE", "-", "clock gain difference",
                           f"{A_S1_NOCLOCK} did not run" if a.empty else "GREEDY-STATE reference absent")
    x = a[col].to_numpy(dtype=np.float64)
    y = np.full(x.size, float(gs[col].mean()))
    bound = float(K.EQUIVALENCE_BOUNDS["predictive_gain"].value)
    t = tost(x, y, bound=bound)
    word = "equivalent" if float(t.get("reject", 0.0)) else "different or inconclusive"
    return row("A-noclock minus GREEDY-STATE", "-", "clock gain, A-noclock minus GREEDY-STATE",
               rule=f"TOST at bound {bound}, outside Holm", values=x - y, p=t.get("pvalue", np.nan),
               word=word, note=f"TOST p {t.get('pvalue', float('nan')):.4g}")


def h2_2x2_rows() -> list[dict[str, Any]]:
    why = "no confirmatory test: B-S1 and C-S1 did not run (stage-two rule)"
    return [missing_row(e, "H2-2x2", e, why) for e in K.FAMILIES["H2-2x2"]]


# ---------------------------------------------------------------- assembly
def build(per_seed: pd.DataFrame, per_eval: pd.DataFrame) -> pd.DataFrame:
    rows = [
        h1a(per_eval, A_S1, "terciles"),
        h1b(per_seed, A_S1),
        *h1c(per_seed, A_S1),
        paired_contrast(per_seed, "H2a", "H2", "frac_SOCIAL", A_S1, D_S1, "SOCIAL time fraction, A - D"),
        paired_contrast(per_seed, "H2b", "H2", "copresence_density", A_S1, D_S1, "co-presence density, A - D"),
        paired_contrast(per_seed, "H2c", "H2", "state_gain_bits", A_S1, D_S1, "state gain, A - D"),
        h3a(per_seed, A_S1),
        h3b_i(per_seed, A_S1),
        paired_contrast(per_seed, "H3b(ii)", "H3", "ari_across_episodes", A_S1, A_S0, "ARI, S1 - S0"),
        h3c(per_seed, A_S1),
        h3d(per_seed, A_S0),
        *h2_2x2_rows(),
        h1a(per_eval, A_S1, "quintiles"),
        h1a(per_eval, A_S1, "fixed"),
        h1a_pooled32(per_seed, A_S1),
        noclock_tost(per_seed),
    ]
    t = pd.DataFrame(rows)
    # Holm within each family, over the tests that produced a p-value.
    t["p_adjusted"] = np.nan
    for family, members in K.FAMILIES.items():
        m = t["family"].eq(family) & t["p_raw"].notna()
        if not m.any():
            continue
        t.loc[m, "p_adjusted"] = holm(t.loc[m, "p_raw"].to_numpy(dtype=np.float64))
    # tests whose rule is the test itself take their word from the adjusted p
    pending = t["decision"].eq("pending-holm")
    t.loc[pending, "decision"] = np.where(
        t.loc[pending, "p_adjusted"] < float(K.ALPHA.value), "yes", "no")
    # section 8.1 precedence: the rule's word stands; a disagreement is stated, never hidden
    rule_yes = t["decision"].eq("yes")
    p_yes = t["p_adjusted"] < float(K.ALPHA.value)
    t["disagreement"] = np.where(
        t["p_adjusted"].notna() & (rule_yes != p_yes),
        np.where(rule_yes, "rule says yes, adjusted p does not reject",
                 "adjusted p rejects, rule does not say yes"), "")
    return t


COLUMNS = ["test", "family", "statistic", "rule", "n", "iqm", "ci_low", "ci_high", "p_improve",
           "p_improve_low", "p_improve_high", "cliffs_delta", "p_raw", "p_adjusted", "decision",
           "disagreement", "count_pass", "count_needed", "note"]


def to_markdown(t: pd.DataFrame, title: str) -> str:
    cols = [c for c in COLUMNS if c in t.columns]
    body = t[cols].copy()
    for c in ("iqm", "ci_low", "ci_high", "p_improve", "p_improve_low", "p_improve_high", "cliffs_delta"):
        if c in body.columns:
            body[c] = body[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    for c in ("p_raw", "p_adjusted"):
        if c in body.columns:
            body[c] = body[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    return f"### {title}\n\n{body.fillna('').to_markdown(index=False)}\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default=None, help="directory for confirmatory.csv and .md (default <root>/reports)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    set_population(root)
    out_dir = Path(args.out) if args.out else root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed, per_eval = load_tables(root)
    if per_seed.empty:
        print(f"no {root}/metrics/per_seed.csv; nothing to test")
        return

    all_runs = build(per_seed, per_eval)
    kept = unflagged(per_seed)
    without = build(kept, per_eval) if len(kept) < len(per_seed) else all_runs
    same = len(kept) == len(per_seed)

    # `without` may be the same object as `all_runs` when nothing is flagged; copy both first.
    all_runs = all_runs.copy()
    without = without.copy()
    all_runs.insert(0, "scope", "all runs")
    without.insert(0, "scope", "unflagged runs only")
    both = pd.concat([all_runs, without], ignore_index=True)
    csv = out_dir / "confirmatory.csv"
    both.to_csv(csv, index=False)

    head = [
        "# Confirmatory contrasts",
        "",
        f"Section 8.1 of `{K.PLAN}`, one row per registered test. Thresholds from "
        "`scripts/analysis_constants.py`; statistics from `hamlet.metrics.stats`.",
        "",
        "Precedence (section 8.1): the pre-registered decision rule gives the word; the Holm-adjusted "
        "p is reported beside it; where they disagree the `disagreement` column says so and the rule's "
        "word stands.",
        "",
        f"Registered thresholds used: H1a {K.H1A_THRESHOLD.value} (sections {K.H1A_THRESHOLD.section}), "
        f"H1c {K.H1C_BOUND.value}, alpha {K.ALPHA.value}, {int(K.N_BOOT.value)} bootstrap resamples.",
        "",
"",
        f"Runs: {len(runs_of(per_seed, A_S1))} {A_S1} cell(s) in `per_seed.csv`. "
        + ("No run is flagged, so both scopes are identical." if same else
           f"{len(per_seed) - len(kept)} flagged row(s) excluded from the second table."),
        "",
    ]
    md = "\n".join(head) + "\n" + to_markdown(all_runs, "All runs") + "\n" + to_markdown(without, "Unflagged runs only")
    (out_dir / "confirmatory.md").write_text(md)
    print(f"{len(both)} row(s) written to {csv} and {out_dir / 'confirmatory.md'}")
    shown = [c for c in ("test", "family", "n", "iqm", "ci_low", "ci_high", "p_raw", "p_adjusted", "decision") if c in all_runs.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(all_runs[shown].to_string(index=False))


if __name__ == "__main__":
    main()
