"""Every registered number the analysis uses, with the section of the plan that fixes it.

Nothing numeric belonging to the pre-registration is written anywhere else: ``confirmatory.py``
and ``make_figures.py`` import from here, so a threshold can be traced to one line and one section.
Values that are properties of the world or the metric library (bin counts, draw counts, capacities)
live in ``hamlet.config`` and ``hamlet.metrics.common``; this module holds only the decision
constants of docs/analysis_plan.md.

Read-only: importing this module changes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

PLAN = "docs/analysis_plan.md"


def root_n_agents(root) -> int:
    """The population size of a run root, read from its cells' own metadata.

    Condition names carry N (``A-S1-N8``), so a script pointed at an archived root must name the
    conditions of that root, not of whatever ``hamlet/config.py`` holds today. Falls back to the
    config only for a root with no trained cells yet. Raises if the root mixes populations.
    """
    import json
    from pathlib import Path
    found = set()
    for md in sorted(Path(root).glob("*/seed*/metadata.json")):
        if "exploratory" in str(md) or "_gate2" in str(md):
            continue
        j = json.loads(md.read_text())
        cfg = j.get("hamlet_config") or j.get("config") or {}
        if cfg.get("n_agents") is not None:
            found.add(int(cfg["n_agents"]))
    if len(found) > 1:
        raise SystemExit(f"{root} mixes populations {sorted(found)}; one root, one study")
    if found:
        return found.pop()
    from hamlet.config import HamletConfig
    return HamletConfig().n_agents


@dataclass(frozen=True)
class Registered:
    """One registered number: its value, the plan section that fixes it, and what it decides."""

    value: float
    section: str
    what: str


# ---- H1 -------------------------------------------------------------------------------------
# The H1a "no effect" threshold for the study the report presents: the N = 8 study. Derived by
# the registered two-clause rule after the state-bin correction: two thirds of GREEDY-CLOCK's
# tercile gain (0.037) is 0.0247, a quarter of the planted effect (0.094) is 0.0235, the rule
# takes the larger and rounds up to the nearest 0.005. Before the state-bin correction it was
# 0.030, and 0.02 under the Manhattan world.
#
# NOT APPLIED HERE. The N = 9 rerun set this threshold on a single anchor, a quarter of the planted
# control's margin over the reference, which reads 0.015. That rule was needed because at N = 9
# the reference's own gain (0.090) had overtaken the plant's margin (0.046), so the two-clause
# rule returned 0.060, a bar the primary positive control could not itself clear. The rerun was
# trained but never analysed, so the report presents the N = 8 study and this module carries the
# N = 8 value. The predictive-gain equivalence bound and the A-noclock TOST bound follow it.
H1A_THRESHOLD = Registered(0.025, "7, 9, 13", "H1a: CI rule and the Wilcoxon shift, bits per tick")
H1A_THRESHOLD_IN_SECTION_8_1 = Registered(0.025, "8.1", "H1a: the Wilcoxon shift, matching section 7")

# H1c: the day-5 drive gap of the shocked run against its paired control must sit below this.
H1C_BOUND = Registered(0.05, "7, 8.1", "H1c: IQM CI upper bound and the Wilcoxon shift, drive units")
# H1c interpretability condition: below this job-time fraction in the A controls, `jobs_closed_d4`
# says nothing and `energy_x2_d4` with `reformation` replaces it, with the substitution logged.
H1C_JOB_FRACTION_MIN = Registered(0.05, "7", "H1c: job time fraction on days 1-5 in the A controls")
H1C_REFORMATION_MIN = Registered(-0.1, "7", "H1c substitute: `reformation` counts as re-formed at or above this")

# H1b and H3a and H3d count rules: the per-seed z or rho must clear its bar in at least this many
# of the eight seeds. Kept as a fraction so a pilot of three seeds scales honestly.
COUNT_RULE_OF_EIGHT = Registered(7, "8.1", "H1b, H3a, H3d: seeds of eight that must clear the bar")
Z_BAR = Registered(1.96, "8.1", "H1b, H3a: the per-seed relabel z bar")

# ---- section 9 equivalence bounds (TOST, two one-sided Wilcoxon tests at alpha 0.05 each) -----
EQUIVALENCE_BOUNDS = {
    # Tied to H1A_THRESHOLD: a threshold move carries the TOST bound with it.
    "predictive_gain": Registered(H1A_THRESHOLD.value, "9", "H1a and the A-noclock minus GREEDY-STATE control"),
    "state_gain": Registered(0.05, "9", "H1b, H2c"),
    "time_fraction": Registered(0.05, "9", "time fractions"),
    "dol_si_density": Registered(0.05, "9", "DOL_indiv, SI, co-presence density"),
    "mean_drive": Registered(0.05, "9", "mean drive"),
}

# ---- section 10, the degeneracy rule ---------------------------------------------------------
# A run is flagged on its stochastic evaluation data if any of these fires. Flagged runs are
# reported and never dropped, and every confirmatory table is shown with and without them.
FLAG_MIN_ZONES_PER_DAY = Registered(3, "10", "few_zones: median distinct symbols per agent-day below this")
FLAG_MAX_SHARE = Registered(0.70, "10", "single_zone: an agent above this share of its ticks in one symbol")
FLAG_MIN_ENTROPY_NATS = Registered(0.05, "10", "low_entropy: mean policy entropy below this")
FLAG_EPISODE_FRACTION = Registered(0.5, "10", "a run is flagged when a flag fires on this share of its episodes")

# ---- test machinery ---------------------------------------------------------------------------
ALPHA = Registered(0.05, "8", "Holm family alpha, and each one-sided TOST test")
N_BOOT = Registered(2000, "6", "stratified bootstrap resamples over seeds")
PLATEAU_TOLERANCE = Registered(0.05, "5", "Gate 2 criterion 3: last five finite checkpoint returns within this of the best")

# ---- Holm families (section 8) -----------------------------------------------------------------
FAMILIES: dict[str, tuple[str, ...]] = {
    "H1": ("H1a", "H1b", "H1c"),
    "H2": ("H2a", "H2b", "H2c"),
    "H3": ("H3a", "H3b(i)", "H3b(ii)", "H3c", "H3d"),
    "H2-2x2": ("reward main effect", "observation main effect", "interaction"),
}
# Outside every family, reported with TOST and no Holm adjustment (section 7, section 8.1 closing).
UNFAMILIED = ("A-noclock minus GREEDY-STATE",)

# ---- human-facing zone names -------------------------------------------------------------------
# The code identifier never changes: CANTEEN is CANTEEN in the zone list, the logs, the metrics, the
# tests, the fixtures and every manifest. These are display names only, for figures, tables, the
# paper and the viewer. The name was "Food Street" for the N = 9 rerun and is "Restaurant" for
# the N = 8 study, matching the rest of the N = 8 text. The viewer prefers the map's own `label`
# property over this table.
DISPLAY_NAMES: dict[str, str] = {
    "HOME": "Home",
    "FARM": "Farm",
    "OFFICE": "Office",
    # "Food Street" was the N = 9 rerun's name. The report presents the N = 8 study, whose figures,
    # tables and replays were all generated reading "Restaurant", so the name stays with the rest
    # of the N = 8 text.
    "CANTEEN": "Restaurant",
    "SOCIAL": "Social",
    "MARKET": "Market",
    "TRANSIT": "Transit",
}


def display(zone: str) -> str:
    """The human-facing name of a zone; unknown names come back unchanged."""
    return DISPLAY_NAMES.get(str(zone).upper(), str(zone))


DECISION_WORDS = ("yes", "no", "inconclusive")


def as_dict() -> dict[str, dict[str, object]]:
    """Every registered constant as ``{name: {value, section, what}}``, for the report headers."""
    out: dict[str, dict[str, object]] = {}
    for name, obj in sorted(globals().items()):
        if isinstance(obj, Registered):
            out[name] = {"value": obj.value, "plan_section": obj.section, "decides": obj.what}
    for key, obj in EQUIVALENCE_BOUNDS.items():
        out[f"EQUIVALENCE_BOUNDS[{key!r}]"] = {"value": obj.value, "plan_section": obj.section, "decides": obj.what}
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(as_dict(), indent=2))
