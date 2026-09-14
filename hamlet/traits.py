"""Six per-agent traits: caps, sampling, population variants and the CSV record.

A trait is read-only after initialisation. The neutral values reproduce the
trait-free world exactly; every other value multiplies or shifts one quantity
of :mod:`hamlet.core` or :mod:`hamlet.policies`:

===============  ===========  =========  =============  ==================================================
trait            type         neutral    hard cap       what it multiplies or shifts
===============  ===========  =========  =============  ==================================================
appetite         multiplier   1.0        [0.5, 2.0]     ``satiety_drain`` per tick (drain only)
metabolism       multiplier   1.0        [0.5, 1.5]     ``energy_drain`` and ``energy_work_drain`` per tick
chronotype       offset (h)   0.0        [-3.0, 3.0]    where the night-rest window sits; GREEDY-CLOCK bedtime
laziness         scalar       0.0        [0, 1]         effort penalty on the reward; greedy act threshold
aptitude         per job      (1.0, 1.0) [0.5, 2.0]     coins per active job tick (geometric mean 1 when normalised)
learning_rate    scalar       0.0        [0, 0.05]      growth of ``skill[job]`` per active tick at that job
===============  ===========  =========  =============  ==================================================

Units: appetite, metabolism and aptitude are dimensionless multipliers;
chronotype is in hours (positive = later); laziness and learning_rate are
dimensionless scalars. ``CAPS`` is the only place the caps are written. A
population variant may narrow a cap but never widen it, and a variant outside
a cap fails at load time; nothing is ever clamped silently.

Randomness: ``sample_traits`` draws agent ``i`` of run seed ``s`` from
``np.random.default_rng([s, TRAITS_SALT, i])`` and never from the world
generator, so changing the population leaves the map, the home assignment,
the initial states, the gossip bit and the admission order untouched.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from hamlet.config import JOBS, N_SCALAR_TRAITS, TRAIT_VECTOR_LEN, TRAITS_SALT, ZONE_NAMES

# ---- caps and neutral values (one place) ----------------------------------------
CAPS: dict[str, tuple[float, float]] = {
    "appetite": (0.5, 2.0),
    "metabolism": (0.5, 1.5),
    "chronotype": (-3.0, 3.0),
    "laziness": (0.0, 1.0),
    "aptitude": (0.5, 2.0),        # per entry of the aptitude vector
    "learning_rate": (0.0, 0.05),
}
NEUTRAL: dict[str, Any] = {
    "appetite": 1.0,
    "metabolism": 1.0,
    "chronotype": 0.0,
    "laziness": 0.0,
    "aptitude": tuple(1.0 for _ in JOBS),
    "learning_rate": 0.0,
}
TRAIT_NAMES = ("appetite", "metabolism", "chronotype", "laziness", "aptitude", "learning_rate")
SCALAR_TRAITS = tuple(t for t in TRAIT_NAMES if t != "aptitude")
JOB_NAMES = tuple(ZONE_NAMES[j] for j in JOBS)
N_JOBS = len(JOBS)
assert len(SCALAR_TRAITS) == N_SCALAR_TRAITS and TRAIT_VECTOR_LEN == N_SCALAR_TRAITS + N_JOBS
# Order of the entries of ``Traits.to_vector``; aptitude contributes one entry per job.
VECTOR_LAYOUT = ("appetite", "metabolism", "chronotype", "laziness") + tuple(
    f"aptitude_{name}" for name in JOB_NAMES
) + ("learning_rate",)
assert len(VECTOR_LAYOUT) == TRAIT_VECTOR_LEN

DIST_TYPES = ("lognormal", "normal", "uniform")
# Columns of traits.csv.
CSV_COLUMNS = ("agent_id", "variant") + SCALAR_TRAITS + tuple(f"aptitude_{name}" for name in JOB_NAMES)
# Draws of a truncated normal are rejected until they fall inside the bounds; this many
# rejections in a row means the bounds exclude nearly all of the mass.
_MAX_REJECTIONS = 10_000


def _unit(value: float, cap: tuple[float, float]) -> float:
    lo, hi = cap
    return (float(value) - lo) / (hi - lo)


def _check_cap(name: str, value: float, cap: tuple[float, float]) -> None:
    lo, hi = cap
    if not np.isfinite(value) or value < lo or value > hi:
        raise ValueError(f"{name} = {value!r} lies outside its cap [{lo}, {hi}]")


# ---- one agent ------------------------------------------------------------------
@dataclass(frozen=True)
class Traits:
    """The six traits of one agent. ``validate`` raises on any value outside ``CAPS``."""

    appetite: float
    metabolism: float
    chronotype: float
    laziness: float
    aptitude: tuple[float, ...]
    learning_rate: float

    @classmethod
    def neutral(cls) -> "Traits":
        return cls(**NEUTRAL)

    def validate(self) -> "Traits":
        for name in SCALAR_TRAITS:
            _check_cap(name, getattr(self, name), CAPS[name])
        if len(self.aptitude) != N_JOBS:
            raise ValueError(f"aptitude needs one entry per job ({N_JOBS}), got {self.aptitude!r}")
        for job, value in zip(JOB_NAMES, self.aptitude):
            _check_cap(f"aptitude[{job}]", value, CAPS["aptitude"])
        return self

    def to_vector(self) -> np.ndarray:
        """``float32 (TRAIT_VECTOR_LEN,)`` with every entry mapped linearly by its cap into [0, 1]."""
        out = [_unit(self.appetite, CAPS["appetite"]), _unit(self.metabolism, CAPS["metabolism"]),
               _unit(self.chronotype, CAPS["chronotype"]), _unit(self.laziness, CAPS["laziness"])]
        out += [_unit(a, CAPS["aptitude"]) for a in self.aptitude]
        out.append(_unit(self.learning_rate, CAPS["learning_rate"]))
        return np.asarray(out, dtype=np.float32)

    def as_row(self, agent_id: int, variant: str) -> dict[str, Any]:
        row: dict[str, Any] = {"agent_id": int(agent_id), "variant": variant}
        for name in SCALAR_TRAITS:
            row[name] = float(getattr(self, name))
        for job, value in zip(JOB_NAMES, self.aptitude):
            row[f"aptitude_{job}"] = float(value)
        return row


# ---- all agents as arrays --------------------------------------------------------
@dataclass
class TraitArrays:
    """The traits of ``N`` agents as arrays: the form :class:`hamlet.core.HamletCore` holds."""

    appetite: np.ndarray       # (N,)
    metabolism: np.ndarray     # (N,)
    chronotype: np.ndarray     # (N,) hours
    laziness: np.ndarray       # (N,)
    aptitude: np.ndarray       # (N, N_JOBS)
    learning_rate: np.ndarray  # (N,)

    @classmethod
    def from_traits(cls, traits: Sequence[Traits]) -> "TraitArrays":
        for t in traits:
            t.validate()
        return cls(
            appetite=np.array([t.appetite for t in traits], dtype=np.float64),
            metabolism=np.array([t.metabolism for t in traits], dtype=np.float64),
            chronotype=np.array([t.chronotype for t in traits], dtype=np.float64),
            laziness=np.array([t.laziness for t in traits], dtype=np.float64),
            aptitude=np.array([t.aptitude for t in traits], dtype=np.float64).reshape(len(traits), N_JOBS),
            learning_rate=np.array([t.learning_rate for t in traits], dtype=np.float64),
        )

    @classmethod
    def neutral(cls, n: int) -> "TraitArrays":
        return cls.from_traits([Traits.neutral()] * n)

    @property
    def n(self) -> int:
        return int(self.appetite.shape[0])

    def as_list(self) -> list[Traits]:
        return [
            Traits(float(self.appetite[i]), float(self.metabolism[i]), float(self.chronotype[i]),
                   float(self.laziness[i]), tuple(float(a) for a in self.aptitude[i]), float(self.learning_rate[i]))
            for i in range(self.n)
        ]

    def validate(self) -> "TraitArrays":
        for t in self.as_list():
            t.validate()
        return self

    def to_vectors(self) -> np.ndarray:
        """``float32 (N, TRAIT_VECTOR_LEN)``, one :meth:`Traits.to_vector` per row."""
        return np.stack([t.to_vector() for t in self.as_list()]) if self.n else np.zeros((0, TRAIT_VECTOR_LEN), np.float32)

    def permuted(self, perm: np.ndarray) -> "TraitArrays":
        """Agent ``i`` of the result carries the traits of agent ``perm[i]``."""
        perm = np.asarray(perm, dtype=np.int64)
        return TraitArrays(**{f.name: getattr(self, f.name)[perm].copy() for f in fields(self)})


# ---- population variants ---------------------------------------------------------
# Schema, per trait (omitted traits are neutral):
#   {"fixed": value}                                            every agent gets value
#   {"dist": {"type": "lognormal", "sigma": s, "clip": [lo, hi]}}   exp(N(0, s)) * neutral, then clipped
#   {"dist": {"type": "normal", "std": s, "clip": [lo, hi]}}        N(neutral, s) truncated to [lo, hi]
#   {"dist": {"type": "uniform", "range": [lo, hi]}}                U(lo, hi)
#   {"groups": [{"fraction": f, "fixed": value}, ...]}          fractions sum to 1; assignment by agent
#                                                               index after a seeded shuffle
# "clip" defaults to the cap and may only narrow it. For aptitude, "fixed" is one value per
# job and a "dist" draws one value per job; the vector is then normalised to geometric mean 1
# when the config says so.
POPULATIONS: dict[str, dict[str, dict[str, Any]]] = {
    "neutral": {},
    "hetero_core": {
        "appetite": {"dist": {"type": "lognormal", "sigma": 0.3}},
        "metabolism": {"dist": {"type": "lognormal", "sigma": 0.3, "clip": [0.5, 1.5]}},
        "chronotype": {"dist": {"type": "normal", "std": 1.5, "clip": [-3.0, 3.0]}},
        "laziness": {"dist": {"type": "uniform", "range": [0.0, 0.6]}},
        "aptitude": {"dist": {"type": "lognormal", "sigma": 0.3}},
        "learning_rate": {"fixed": 0.01},
    },
    # Stress variant, not a positive control: half the population burns energy at the cap.
    "metabolism_split": {
        "metabolism": {"groups": [{"fraction": 0.5, "fixed": 1.0}, {"fraction": 0.5, "fixed": 1.5}]},
    },
    # Positive control for routine (H1): two bedtimes four hours apart.
    "chronotype_split": {
        "chronotype": {"groups": [{"fraction": 0.5, "fixed": -2.0}, {"fraction": 0.5, "fixed": 2.0}]},
    },
    # Positive control for the division of labour (H3): half prefer FARM, half OFFICE.
    "aptitude_split": {
        "aptitude": {"groups": [{"fraction": 0.5, "fixed": [1.4, 0.7]}, {"fraction": 0.5, "fixed": [0.7, 1.4]}]},
    },
    "aptitude_only": {
        "aptitude": {"dist": {"type": "lognormal", "sigma": 0.3}},
    },
    "learning_only": {
        "learning_rate": {"fixed": 0.02},
    },
}


@dataclass(frozen=True)
class Population:
    """A validated variant: ``name`` and the per-trait specification (see the schema above)."""

    name: str
    spec: Mapping[str, Mapping[str, Any]]

    def entry(self, trait: str) -> Optional[Mapping[str, Any]]:
        return self.spec.get(trait)


PopulationLike = Union[str, Population, Mapping[str, Mapping[str, Any]]]


def _as_pair(value: Any, what: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a pair [lo, hi], got {value!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError(f"{what} must satisfy lo <= hi, got {value!r}")
    return lo, hi


def _within(inner: tuple[float, float], cap: tuple[float, float], what: str) -> None:
    if inner[0] < cap[0] or inner[1] > cap[1]:
        raise ValueError(f"{what} {list(inner)} widens the cap {list(cap)}; a variant may only narrow it")


def _check_fixed(trait: str, value: Any) -> None:
    cap = CAPS[trait]
    if trait == "aptitude":
        vals = list(value) if isinstance(value, (list, tuple, np.ndarray)) else None
        if vals is None or len(vals) != N_JOBS:
            raise ValueError(f"aptitude fixed value needs {N_JOBS} entries (one per job), got {value!r}")
        for job, v in zip(JOB_NAMES, vals):
            _check_cap(f"aptitude[{job}]", float(v), cap)
    else:
        if isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"{trait} fixed value must be a scalar, got {value!r}")
        _check_cap(trait, float(value), cap)


def _check_dist(trait: str, dist: Mapping[str, Any]) -> None:
    cap = CAPS[trait]
    kind = dist.get("type")
    if kind not in DIST_TYPES:
        raise ValueError(f"{trait}: unknown distribution type {kind!r}; choose from {DIST_TYPES}")
    clip = _as_pair(dist["clip"], f"{trait} clip") if "clip" in dist else cap
    _within(clip, cap, f"{trait} clip")
    if kind == "lognormal":
        if trait in ("chronotype", "laziness", "learning_rate"):
            raise ValueError(f"{trait} is not a multiplier; use a normal or uniform distribution")
        if float(dist.get("sigma", -1.0)) < 0:
            raise ValueError(f"{trait}: lognormal needs sigma >= 0")
    elif kind == "normal":
        if float(dist.get("std", -1.0)) < 0:
            raise ValueError(f"{trait}: normal needs std >= 0")
        mean = float(dist.get("mean", NEUTRAL[trait] if trait != "aptitude" else 1.0))
        if not clip[0] <= mean <= clip[1]:
            raise ValueError(f"{trait}: the normal mean {mean} lies outside the clip {list(clip)}")
    else:
        rng_ = _as_pair(dist.get("range"), f"{trait} range")
        _within(rng_, cap, f"{trait} range")
    unknown = set(dist) - {"type", "sigma", "std", "mean", "range", "clip"}
    if unknown:
        raise ValueError(f"{trait}: unknown distribution keys {sorted(unknown)}")


def load_population(population: PopulationLike) -> Population:
    """Resolve a name (in ``POPULATIONS``), a mapping or a :class:`Population`; raise if invalid.

    Every fixed value, clip and range is checked against ``CAPS`` here, so an
    invalid variant fails before a single agent is drawn.
    """
    if isinstance(population, Population):
        name, spec = population.name, population.spec
    elif isinstance(population, str):
        if population not in POPULATIONS:
            raise ValueError(f"unknown population {population!r}; choose from {sorted(POPULATIONS)}")
        name, spec = population, POPULATIONS[population]
    elif isinstance(population, Mapping):
        name, spec = str(population.get("name", "custom")), {k: v for k, v in population.items() if k != "name"}
    else:
        raise TypeError(f"population must be a name, a mapping or a Population, got {type(population).__name__}")
    for trait, entry in spec.items():
        if trait not in TRAIT_NAMES:
            raise ValueError(f"population {name!r}: unknown trait {trait!r}; the six traits are {TRAIT_NAMES}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"population {name!r}: {trait} must map to a dict, got {entry!r}")
        keys = [k for k in ("fixed", "dist", "groups") if k in entry]
        if len(keys) != 1 or set(entry) != {keys[0]}:
            raise ValueError(f"population {name!r}: {trait} needs exactly one of fixed, dist or groups, got {sorted(entry)}")
        if "fixed" in entry:
            _check_fixed(trait, entry["fixed"])
        elif "dist" in entry:
            _check_dist(trait, entry["dist"])
        else:
            groups = entry["groups"]
            if not groups:
                raise ValueError(f"population {name!r}: {trait} groups must not be empty")
            total = 0.0
            for g in groups:
                if set(g) != {"fraction", "fixed"}:
                    raise ValueError(f"population {name!r}: each {trait} group needs fraction and fixed, got {sorted(g)}")
                if not 0.0 <= float(g["fraction"]) <= 1.0:
                    raise ValueError(f"population {name!r}: {trait} group fraction {g['fraction']!r} not in [0, 1]")
                total += float(g["fraction"])
                _check_fixed(trait, g["fixed"])
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"population {name!r}: {trait} group fractions sum to {total}, not 1")
    return Population(name, {k: dict(v) for k, v in spec.items()})


def agent_rng(run_seed: int, agent_id: int) -> np.random.Generator:
    """The trait generator of one agent: ``default_rng([run_seed, TRAITS_SALT, agent_id])``."""
    return np.random.default_rng([int(run_seed), TRAITS_SALT, int(agent_id)])


def _group_rng(run_seed: int, trait_index: int) -> np.random.Generator:
    """Shuffle generator of one group assignment; a four-entry key never collides with an agent key."""
    return np.random.default_rng([int(run_seed), TRAITS_SALT, TRAITS_SALT, int(trait_index)])


def _group_counts(fractions: Sequence[float], n: int) -> list[int]:
    edges = [int(np.floor(c * n + 0.5)) for c in np.cumsum([float(f) for f in fractions])]
    edges[-1] = n
    counts, last = [], 0
    for e in edges:
        counts.append(max(e - last, 0))
        last = max(e, last)
    return counts


def _draw_scalar(rng: np.random.Generator, dist: Mapping[str, Any], trait: str, neutral: float) -> float:
    cap = CAPS[trait]
    clip = _as_pair(dist["clip"], f"{trait} clip") if "clip" in dist else cap
    kind = dist["type"]
    if kind == "lognormal":
        return float(np.clip(neutral * np.exp(rng.normal(0.0, float(dist["sigma"]))), *clip))
    if kind == "normal":
        mean, std = float(dist.get("mean", neutral)), float(dist["std"])
        for _ in range(_MAX_REJECTIONS):
            x = float(rng.normal(mean, std))
            if clip[0] <= x <= clip[1]:
                return x
        raise RuntimeError(f"{trait}: truncated normal rejected {_MAX_REJECTIONS} draws in a row")
    lo, hi = _as_pair(dist["range"], f"{trait} range")
    return float(rng.uniform(lo, hi))


def normalise_aptitude(aptitude: Sequence[float]) -> tuple[float, ...]:
    """Rescale an aptitude vector to geometric mean 1 (its ratios are preserved)."""
    a = np.asarray(aptitude, dtype=np.float64)
    g = float(np.exp(np.log(a).mean()))
    return tuple(float(v) for v in a / g)


def sample_traits(
    population: PopulationLike,
    n_agents: int,
    run_seed: int,
    normalise: bool = True,
) -> list[Traits]:
    """One :class:`Traits` per agent for ``population`` under ``run_seed``.

    Draws for agent ``i`` come from :func:`agent_rng` ``(run_seed, i)`` in
    ``TRAIT_NAMES`` order, so the pair ``(run_seed, agent_id)`` fixes an agent's
    traits whatever ``n_agents`` is. Group memberships are dealt out in agent
    index order after a shuffle seeded from the run seed alone. With
    ``normalise`` every aptitude vector is rescaled to geometric mean 1. Every
    result is validated against ``CAPS``.
    """
    pop = load_population(population)
    n = int(n_agents)
    values: dict[str, list[Any]] = {t: [NEUTRAL[t]] * n for t in TRAIT_NAMES}
    for k, trait in enumerate(TRAIT_NAMES):
        entry = pop.entry(trait)
        if entry is None:
            continue
        if "fixed" in entry:
            v = tuple(float(x) for x in entry["fixed"]) if trait == "aptitude" else float(entry["fixed"])
            values[trait] = [v] * n
        elif "groups" in entry:
            order = _group_rng(run_seed, k).permutation(n)
            counts = _group_counts([g["fraction"] for g in entry["groups"]], n)
            start = 0
            for g, c in zip(entry["groups"], counts):
                v = tuple(float(x) for x in g["fixed"]) if trait == "aptitude" else float(g["fixed"])
                for i in order[start:start + c]:
                    values[trait][int(i)] = v
                start += c
    for i in range(n):
        rng = agent_rng(run_seed, i)
        for trait in TRAIT_NAMES:
            entry = pop.entry(trait)
            if entry is None or "dist" not in entry:
                continue
            if trait == "aptitude":
                values[trait][i] = tuple(_draw_scalar(rng, entry["dist"], trait, 1.0) for _ in JOBS)
            else:
                values[trait][i] = _draw_scalar(rng, entry["dist"], trait, float(NEUTRAL[trait]))
    out = []
    for i in range(n):
        apt = tuple(values["aptitude"][i])
        if normalise:
            apt = normalise_aptitude(apt)
        out.append(Traits(
            appetite=values["appetite"][i], metabolism=values["metabolism"][i], chronotype=values["chronotype"][i],
            laziness=values["laziness"][i], aptitude=apt, learning_rate=values["learning_rate"][i],
        ).validate())
    return out


# ---- CSV ----------------------------------------------------------------------------
def write_traits_csv(path: Path, traits: Sequence[Traits], population: str) -> Path:
    """Write one row per agent: agent_id, variant, the five scalar traits, aptitude per job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for i, t in enumerate(traits):
            writer.writerow(t.as_row(i, population))
    return path


def read_traits_csv(path: Path) -> tuple[list[Traits], str]:
    """Read a file written by :func:`write_traits_csv`; returns ``(traits in agent order, variant)``."""
    with Path(path).open(newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: int(r["agent_id"]))
    traits = [
        Traits(
            appetite=float(r["appetite"]), metabolism=float(r["metabolism"]), chronotype=float(r["chronotype"]),
            laziness=float(r["laziness"]), aptitude=tuple(float(r[f"aptitude_{job}"]) for job in JOB_NAMES),
            learning_rate=float(r["learning_rate"]),
        )
        for r in rows
    ]
    variant = rows[0]["variant"] if rows else "neutral"
    return traits, variant
