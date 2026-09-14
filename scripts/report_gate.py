"""Competence-gate and calibration report for the hand-written schedulers.

Every number quoted about the greedy baselines (night-rule decomposition,
hunger time constant, calibration sensitivity, canteen load, queue statistics,
per-day gate values) is produced by this script and nothing else. Run::

    uv run python scripts/report_gate.py                 # all sections, current config
    uv run python scripts/report_gate.py --section rules --satiety-drain 0.004
    uv run python scripts/report_gate.py --section load --population metabolism_split

Sections
  gate         GREEDY-CLOCK on seeds 0, 1, 2, 10000, 10001, 10002: gate mean and minimum
  rules        night-rule decomposition (literal / hysteresis / hysteresis_wake) vs GREEDY-STATE
               and GREEDY-CLOCK-WORK
  sensitivity  gate under satiety_drain x canteen seats for each night rule
  load         daytime canteen load factor and gate under population variants
  horizon      per-day gate, coin balance, 06:00 queue statistics over the whole episode
  calibrate    gate per population under the current calibration
  positive     DOL_indiv and its null z under GREEDY-CLOCK per population
  discrepancy  the same rule measured with two seed sets and two aggregation conventions

Population variants are the real trait variants of ``hamlet/traits.py``
(``HamletConfig.population``); the traits of a seed are drawn by the core from
that seed and never touch the world randomness. The gate is an absolute 0.90
of agent-ticks (days after burn-in) on which every need exceeds 0.3; it was
fixed after observing baseline runs and is frozen, and it is defined on the
neutral population (the other populations are descriptive).
The gate is judged on the minimum over seeds; this script asserts nothing and
prints the minimum next to the mean.

Outputs: markdown tables on stdout and CSV copies under runs/reports/.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hamlet.config import CANTEEN, HOME, JOBS, MARKET, GREEDY_NIGHT_RULES, HamletConfig
from hamlet.core import HamletCore
from hamlet.policies import GreedyClockPolicy, GreedyClockWorkPolicy, GreedyStatePolicy
from hamlet.traits import POPULATIONS

GATE = 0.90
NEED_FLOOR = 0.3
POSITIVE_CONTROL_ARI_MIN = 0.5   # aptitude_split cross-episode ARI, every seed (set after observation, frozen)
NEUTRAL_ARI_MAX = 0.2            # neutral cross-episode ARI, every seed
DEFAULT_SEEDS = (10000, 10001, 10002)
GATE_SEEDS = (0, 1, 2, 10000, 10001, 10002)
OUT_DIR = Path("runs/reports")

# ---- population variants -----------------------------------------------------------
def population_config(cfg: HamletConfig, population: str) -> HamletConfig:
    """The same world with another trait variant; ``HamletConfig.validate`` rejects unknown names."""
    return dataclasses.replace(cfg, population=population)


# ---- simulation -------------------------------------------------------------------
WORK_ABOVE: list[float] = []   # set from --work-above; empty means the config default


def make_policy(name: str):
    """A fresh scheduler; GREEDY-CLOCK-WORK reads its aptitudes from the core's traits."""
    if name == "GREEDY-STATE":
        return GreedyStatePolicy()
    if name == "GREEDY-CLOCK-WORK":
        kw = {"work_above": WORK_ABOVE[0]} if WORK_ABOVE else {}
        return GreedyClockWorkPolicy(**kw)
    if name.startswith("GREEDY-CLOCK"):
        rule = name.split(":")[1] if ":" in name else None
        return GreedyClockPolicy() if rule is None else GreedyClockPolicy(night_rule=rule)
    raise ValueError(name)


def simulate(cfg: HamletConfig, policy, seed: int) -> dict[str, np.ndarray]:
    core = HamletCore(cfg, seed)
    obs = core.reset(seed)
    T = cfg.episode_ticks
    n = cfg.N
    rec = {k: np.zeros((T, n)) for k in ("E", "F", "C", "D", "W", "coins_earned")}
    rec.update({k: np.zeros((T, n), dtype=np.int64) for k in ("zone",)})
    rec.update({k: np.zeros((T, n), dtype=bool) for k in ("active", "queued")})
    for t in range(T):
        a, _ = policy.act(obs, core)
        obs, _, _ = core.step(a)
        for k in ("E", "F", "C", "D", "W", "coins_earned", "zone", "active", "queued"):
            rec[k][t] = getattr(core, k)
    rec["t"] = np.arange(T)
    rec["day"] = rec["t"] // cfg.ticks_per_day
    rec["t_day"] = rec["t"] % cfg.ticks_per_day
    rec["traits"] = core.traits
    return rec


def after_burn_in(rec, cfg):
    m = rec["day"] >= cfg.burn_in_days
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in rec.items()}


def gate_value(rec, cfg) -> float:
    r = after_burn_in(rec, cfg)
    ok = (r["E"] >= NEED_FLOOR) & (r["F"] >= NEED_FLOOR) & (r["C"] >= NEED_FLOOR)
    return float(ok.mean())


def hunger_clock(rec, cfg) -> float:
    """Empirical hours from leaving a feeding zone at F >= 0.85 to F < 0.3 without eating."""
    r = after_burn_in(rec, cfg)
    eat = r["active"] & np.isin(r["zone"], [CANTEEN, MARKET])
    hours = []
    for i in range(cfg.N):
        f, e = r["F"][:, i], eat[:, i]
        t0 = None
        for k in range(1, len(f)):
            if e[k - 1] and not e[k] and f[k] >= 0.85:
                t0 = k
            elif t0 is not None and e[k]:
                t0 = None
            if t0 is not None and f[k] < NEED_FLOOR:
                hours.append((k - t0) * 6 / 60)
                t0 = None
    return float(np.mean(hours)) if hours else float("nan")


def rule_row(rec, cfg) -> dict[str, float]:
    r = after_burn_in(rec, cfg)
    night = np.array([cfg.is_night(int(td)) for td in r["t_day"]])
    home = (r["zone"] == HOME) & r["active"]
    job = np.isin(r["zone"], JOBS) & r["active"]
    return {
        "gate": gate_value(rec, cfg),
        "E>=0.3": float((r["E"] >= NEED_FLOOR).mean()),
        "F>=0.3": float((r["F"] >= NEED_FLOOR).mean()),
        "C>=0.3": float((r["C"] >= NEED_FLOOR).mean()),
        "mean_D": float(r["D"].mean()),
        "night_E<0.3": float((r["E"][night] < NEED_FLOOR).mean()),
        "night_F<0.3": float((r["F"][night] < NEED_FLOOR).mean()),
        "night_C<0.3": float((r["C"][night] < NEED_FLOOR).mean()),
        "F_at_0600": float(r["F"][r["t_day"] == cfg.night[1]].mean()),
        "home_ticks_at_night": float(home[night].mean()),
        "sleep_ticks_per_agent_day": float(home.mean() * cfg.ticks_per_day),
        "job_ticks_per_agent_day": float(job.mean() * cfg.ticks_per_day),
        "coins_per_agent_day": float(r["coins_earned"].mean() * cfg.ticks_per_day),
        "hours_meal_to_0.3": hunger_clock(rec, cfg),
    }


# ---- sections ---------------------------------------------------------------------
def section_gate(cfg: HamletConfig, populations: Iterable[str] = ("neutral",), seeds: Iterable[int] = GATE_SEEDS) -> pd.DataFrame:
    """GREEDY-CLOCK (the reference definition) on the gate seeds: one row per (population, seed)."""
    rows = []
    for pop in populations:
        c = population_config(cfg, pop)
        for seed in seeds:
            rec = simulate(c, make_policy("GREEDY-CLOCK"), seed)
            rows.append({"population": pop, "policy": "GREEDY-CLOCK", "seed": seed, "gate": gate_value(rec, c)})
    return pd.DataFrame(rows)


def section_rules(cfg: HamletConfig, seeds: Iterable[int]) -> pd.DataFrame:
    rows = []
    names = [f"GREEDY-CLOCK:{r}" for r in GREEDY_NIGHT_RULES] + ["GREEDY-STATE", "GREEDY-CLOCK-WORK"]
    for seed in seeds:
        for name in names:
            row = rule_row(simulate(cfg, make_policy(name), seed), cfg)
            rows.append({"policy": name, "seed": seed, **row})
    return pd.DataFrame(rows)


def section_sensitivity(cfg: HamletConfig, seeds: Iterable[int]) -> pd.DataFrame:
    rows = []
    for drain in (0.004, 0.003):
        for seats in (2, 3):
            c = dataclasses.replace(cfg, satiety_drain=drain, cap_canteen=seats)
            for seed in seeds:
                for rule in GREEDY_NIGHT_RULES:
                    g = gate_value(simulate(c, make_policy(f"GREEDY-CLOCK:{rule}"), seed), c)
                    rows.append({"satiety_drain": drain, "canteen_seats": seats, "rule": rule, "seed": seed, "gate": g})
                g = gate_value(simulate(c, make_policy("GREEDY-STATE"), seed), c)
                rows.append({"satiety_drain": drain, "canteen_seats": seats, "rule": "GREEDY-STATE", "seed": seed, "gate": g})
    df = pd.DataFrame(rows)
    return df.groupby(["satiety_drain", "canteen_seats", "rule"])["gate"].agg(["mean", "min"]).reset_index()


def canteen_load(rec, cfg) -> dict[str, float]:
    """Daytime (06:00-22:00) canteen seat-ticks demanded over seat-ticks available."""
    r = after_burn_in(rec, cfg)
    day = (r["t_day"] >= cfg.night[1]) & (r["t_day"] < cfg.night[0])
    at_canteen = (r["zone"] == CANTEEN) & (r["active"] | r["queued"])
    demanded = float(at_canteen[day].sum())
    n_days = cfg.n_days - cfg.burn_in_days
    available = float(cfg.capacity(CANTEEN) * (cfg.night[0] - cfg.night[1]) * n_days)
    served = float((at_canteen & r["active"])[day].sum())
    return {"load_observed": demanded / available, "seat_utilisation": served / available}


def analytic_load(cfg: HamletConfig, appetite: np.ndarray) -> float:
    daily_need = cfg.ticks_per_day * cfg.satiety_drain * appetite          # per agent per day
    net_gain = cfg.satiety_canteen_gain - cfg.satiety_drain * appetite     # per canteen tick
    ticks_needed = (daily_need / net_gain).sum()
    return float(ticks_needed / (cfg.capacity(CANTEEN) * (cfg.night[0] - cfg.night[1])))


def section_load(cfg: HamletConfig, seeds: Iterable[int], populations: Iterable[str]) -> pd.DataFrame:
    rows = []
    for pop in populations:
        c = population_config(cfg, pop)
        for seed in seeds:
            for name in ("GREEDY-CLOCK", "GREEDY-STATE"):
                rec = simulate(c, make_policy(name), seed)
                traits = rec["traits"]
                rows.append({
                    "population": pop, "policy": name, "seed": seed,
                    "gate": gate_value(rec, c), **canteen_load(rec, c),
                    "load_analytic": analytic_load(c, traits.appetite),
                    "mean_appetite": float(traits.appetite.mean()), "mean_metabolism": float(traits.metabolism.mean()),
                })
    return pd.DataFrame(rows)


def section_horizon(cfg: HamletConfig, seeds: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_day, queue = [], []
    wake = cfg.night[1]
    for seed in seeds:
        rec = simulate(cfg, make_policy("GREEDY-CLOCK"), seed)
        for d in range(cfg.n_days):
            m = rec["day"] == d
            ok = (rec["E"][m] >= NEED_FLOOR) & (rec["F"][m] >= NEED_FLOOR) & (rec["C"][m] >= NEED_FLOOR)
            per_day.append({"seed": seed, "day": d, "gate": float(ok.mean()), "mean_coins": float(rec["W"][m].mean()),
                            "mean_F": float(rec["F"][m].mean()), "mean_E": float(rec["E"][m].mean())})
            # 06:00 queue: from wake until noon, per agent
            morning = m & (rec["t_day"] >= wake) & (rec["t_day"] < wake + 60)
            q = rec["queued"][morning]                       # (60, N)
            a = ((rec["zone"] == CANTEEN) & rec["active"])[morning]
            longest = 0
            for i in range(cfg.N):
                run, best = 0, 0
                for v in q[:, i]:
                    run = run + 1 if v else 0
                    best = max(best, run)
                longest = max(longest, best)
            served_by_hour = {}
            for h in range(1, 7):
                served_by_hour[f"served_by_{(wake + 10 * h) // 10:02d}00"] = int(a[: 10 * h].any(axis=0).sum())
            queue.append({"seed": seed, "day": d, "max_wait_ticks": longest, "max_wait_hours": longest * 6 / 60,
                          "queued_agent_ticks_0600_1200": int(q.sum()), **served_by_hour})
    return pd.DataFrame(per_day), pd.DataFrame(queue)


def section_calibrate(cfg: HamletConfig, seeds: Iterable[int], populations: Iterable[str]) -> pd.DataFrame:
    """Gate under GREEDY-CLOCK per population at the current calibration.

    The candidate list once held single-parameter alternatives (energy_rest_gain
    0.022 to 0.030, night_rest_bonus 1.0, metabolism caps 1.4 and 1.5). The
    decision was to keep the world as it is and narrow the metabolism cap to
    (0.5, 1.5) (``hamlet.traits.CAPS``), so only the "as is" row remains.
    """
    candidates = [
        ("as is", {}),
    ]
    rows = []
    for label, overrides in candidates:
        base = dataclasses.replace(cfg, **overrides)
        for pop in populations:
            c = population_config(base, pop)
            for seed in seeds:
                g = gate_value(simulate(c, make_policy("GREEDY-CLOCK"), seed), c)
                rows.append({"calibration": label, "population": pop, "seed": seed, "gate": g})
    df = pd.DataFrame(rows)
    return df.groupby(["calibration", "population"], sort=False)["gate"].agg(["mean", "min"]).reset_index()


POSITIVE_TRAITS_SEED = 0   # one trait assignment for every evaluation seed of the positive-control section


def section_positive(cfg: HamletConfig, seeds: Iterable[int], populations: Iterable[str]) -> pd.DataFrame:
    """DOL_indiv (agent-day shuffle null) and cross-seed ARI under GREEDY-CLOCK-WORK per population.

    The DOL positive control is aptitude_split: half the agents prefer FARM,
    half OFFICE (normalised aptitudes). It must move DOL clearly above the
    neutral range with a high cross-episode ARI; neutral must not.
    metabolism_split is a stress variant and is reported for the record.
    Traits are a property of the run, not of the evaluation episode: one
    assignment (trait seed ``POSITIVE_TRAITS_SEED``) is used for every
    evaluation seed, so the cross-seed ARI is a cross-episode ARI. The
    ``--work-above`` override of the work rule applies here.
    """
    from hamlet.evaluate import rollout
    from hamlet.metrics.specialisation import ari, cluster_agents, dol_indiv, time_budget

    rows = []
    labels_by_pop: dict[str, list[np.ndarray]] = {}
    job_labels_by_pop: dict[str, list[np.ndarray]] = {}
    for pop in populations:
        c = population_config(cfg, pop)
        for seed in seeds:
            policy = make_policy("GREEDY-CLOCK-WORK")
            df = rollout(c, policy, seed, 0, traits_seed=POSITIVE_TRAITS_SEED)
            counts = time_budget(df, per_day=True)
            res = dol_indiv(counts, n_perm=200, rng=np.random.default_rng(seed))
            # secondary, descriptive: DOL over the two job zones only (how work, not time, is divided)
            job_counts = counts[:, :, list(JOBS)]
            dol_jobs = dol_indiv(job_counts, n_perm=200, rng=np.random.default_rng(seed + 1))
            labels, sil = cluster_agents(counts.sum(axis=1), max_k=4)
            labels_by_pop.setdefault(pop, []).append(np.asarray(labels))
            job_labels, _ = cluster_agents(job_counts.sum(axis=1), max_k=4)
            job_labels_by_pop.setdefault(pop, []).append(np.asarray(job_labels))
            rows.append({"population": pop, "seed": seed, "dol_indiv": res["value"], "z_dayshuffle": res["z"],
                         "dol_jobs_secondary": dol_jobs["value"], "dol_jobs_z": dol_jobs["z"],
                         "n_clusters": int(len(set(labels.tolist()))), "silhouette": float(sil)})
    out = pd.DataFrame(rows)
    # cross-episode ARI: the same agent indices carry the same traits in every evaluation seed.
    # Per seed: the mean ARI of that seed's cluster labels against every other seed's labels,
    # on all-zone clusters (the ruler) and, as a secondary descriptive column, on job-zone clusters.
    def per_seed_ari(labs: list[np.ndarray]) -> list[float]:
        vals = []
        for i in range(len(labs)):
            others = [ari(labs[i], labs[j]) for j in range(len(labs)) if j != i]
            vals.append(float(np.mean(others)) if others else float("nan"))
        return vals
    ari_all = {pop: per_seed_ari(labs) for pop, labs in labels_by_pop.items()}
    ari_jobs = {pop: per_seed_ari(labs) for pop, labs in job_labels_by_pop.items()}
    out["ari_across_seeds"] = [ari_all[p][i] for p, i in zip(out["population"], out.groupby("population").cumcount())]
    out["ari_jobs_secondary"] = [ari_jobs[p][i] for p, i in zip(out["population"], out.groupby("population").cumcount())]
    # Positive-control criterion (set after observation, then frozen): aptitude_split ARI >= 0.5
    # on every seed and neutral ARI <= 0.2 on every seed, both on all-zone clusters.
    crit = {"aptitude_split": lambda v: v >= POSITIVE_CONTROL_ARI_MIN, "neutral": lambda v: v <= NEUTRAL_ARI_MAX}
    out["criterion"] = [("pass" if crit[p](v) else "FAIL") if p in crit else "" for p, v in zip(out["population"], out["ari_across_seeds"])]
    return out


def section_discrepancy(cfg: HamletConfig) -> pd.DataFrame:
    """Rule hysteresis_wake on two seed sets, at the pre- and post-calibration hunger decay.

    Records the cause of a quoted 94-97% that could not be reproduced on the
    evaluation seeds: the same rule and code on seeds 0-2 versus 10000-10002.
    """
    rows = []
    for drain in (0.004, cfg.satiety_drain):
        c = dataclasses.replace(cfg, satiety_drain=drain)
        for seed_set, seeds in (("0-2", (0, 1, 2)), ("evaluation", DEFAULT_SEEDS)):
            gates = [gate_value(simulate(c, make_policy("GREEDY-CLOCK:hysteresis_wake"), s), c) for s in seeds]
            rows.append({"satiety_drain": drain, "seed_set": seed_set, "seeds": str(seeds),
                         "gate_mean": float(np.mean(gates)), "gate_min": float(np.min(gates)),
                         "gate_max": float(np.max(gates))})
    return pd.DataFrame(rows)


# ---- main -------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=["all", "gate", "rules", "sensitivity", "load", "horizon", "calibrate", "positive", "discrepancy"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--satiety-drain", type=float, default=None)
    ap.add_argument("--canteen-seats", type=int, default=None)
    ap.add_argument("--population", nargs="+", default=["neutral", "hetero_core", "metabolism_split", "aptitude_split", "chronotype_split"],
                    choices=sorted(POPULATIONS), help="trait variants (hamlet/traits.py)")
    ap.add_argument("--work-above", type=float, default=None, help="GREEDY-CLOCK-WORK threshold override (sensitivity only)")
    args = ap.parse_args(argv)

    cfg = HamletConfig()
    if args.work_above is not None:
        WORK_ABOVE[:] = [args.work_above]
    if args.satiety_drain is not None:
        cfg = dataclasses.replace(cfg, satiety_drain=args.satiety_drain)
    if args.canteen_seats is not None:
        cfg = dataclasses.replace(cfg, cap_canteen=args.canteen_seats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    night_len = (cfg.ticks_per_day - cfg.night[0]) + cfg.night[1]
    print(f"config: satiety_drain={cfg.satiety_drain} canteen_seats={cfg.capacity(CANTEEN)} gate={GATE} floor={NEED_FLOOR}")
    print(f"hunger clock: 1.0->0.3 = {(1 - NEED_FLOOR) / cfg.satiety_drain:.0f} ticks = "
          f"{(1 - NEED_FLOOR) / cfg.satiety_drain / 10:.1f} h; night = {night_len} ticks = {night_len / 10:.0f} h, "
          f"costing {night_len * cfg.satiety_drain:.2f} satiety\n")

    def emit(name: str, df: pd.DataFrame) -> None:
        print(f"## {name}\n{df.round(3).to_markdown(index=False)}\n")
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    if args.section in ("all", "gate"):
        df = section_gate(cfg, args.population)
        emit("gate_per_seed", df)
        for pop, part in df.groupby("population", sort=False):
            note = "the gate is defined here" if pop == "neutral" else "descriptive only; the gate is defined on neutral"
            print(f"GREEDY-CLOCK gate, population {pop}, seeds {list(GATE_SEEDS)}: mean {part['gate'].mean():.4f}, "
                  f"min {part['gate'].min():.4f} (threshold {GATE}, judged on the minimum; {note})")
        print()
    if args.section in ("all", "rules"):
        df = section_rules(cfg, args.seeds)
        emit("rules_per_seed", df)
        emit("rules_mean", df.drop(columns="seed").groupby("policy", sort=False).mean().reset_index())
        emit("rules_min_gate", df.groupby("policy", sort=False)["gate"].agg(["mean", "min"]).reset_index())
    if args.section in ("all", "sensitivity"):
        emit("sensitivity", section_sensitivity(cfg, args.seeds))
    if args.section in ("all", "load"):
        df = section_load(cfg, args.seeds, args.population)
        emit("load_per_seed", df)
        emit("load_mean", df.drop(columns="seed").groupby(["population", "policy"], sort=False).agg(["mean", "min"]).reset_index())
    if args.section in ("all", "horizon"):
        per_day, queue = section_horizon(cfg, args.seeds)
        emit("horizon_per_day", per_day.groupby("day").mean().drop(columns="seed").reset_index())
        emit("horizon_queue", queue.groupby("day").agg(["mean", "max"]).drop(columns="seed").reset_index())
    if args.section in ("all", "calibrate"):
        emit("calibrate", section_calibrate(cfg, args.seeds, args.population))
    if args.section in ("all", "positive"):
        emit("positive_control", section_positive(cfg, args.seeds, args.population))
    if args.section in ("all", "discrepancy"):
        emit("discrepancy", section_discrepancy(cfg))


if __name__ == "__main__":
    main()
