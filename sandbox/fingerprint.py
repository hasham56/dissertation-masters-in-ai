"""Behavioural fingerprint of one evaluation Parquet: a JSON dictionary and one PNG actogram.

Command line::

    uv run python -m sandbox.fingerprint PARQUET --out DIR [--agent K]
    uv run python -m sandbox.fingerprint PARQUET --diff OTHER_PARQUET --out DIR [--agent K]

``PARQUET`` is one evaluation episode in ``hamlet.config.LOG_COLUMNS`` (a
sandbox rollout under ``sandbox/runs/`` or, read-only, a confirmatory-arm file
under ``runs/``); ``DIR`` must lie inside ``sandbox/runs/``. Nothing is
ever written beside the input.

The fingerprint (``fingerprint(df)``), days ``>= burn_in_days`` only, every
number from the existing ``hamlet.metrics`` functions:

- ``zone_fraction``: time fraction per zone (HOME .. TRANSIT), the sum over
  agents of ``specialisation.time_budget(df)`` normalised; ``zone_fraction_per_agent``
  the same per agent;
- ``hour_zone``: the (8, 7) hour-bin x zone occupancy matrix, fraction of
  agent-ticks, from the ``hour_bin`` column of ``routine.add_bins(df)``;
- ``mean_state``: mean E, F, C over agents and ticks; ``mean_drive``: mean D as
  logged; ``mean_drive_per_need``: the per-need terms of the study's drive,
  ``mean((1 - x_i)^2)``, so a run with other set points is reported on the
  study's ruler as well;
- ``dol_indiv``: within-episode division of labour with its agent-day-shuffle
  z, ``specialisation.dol_indiv(specialisation.time_budget(df, per_day=True))``;
- ``source``: ``condition``, ``seed`` and ``checkpoint`` of the file and
  ``{"exploratory": true}``.

``fingerprint_diff(a, b)`` returns the deltas (b minus a) of every scalar and
vector entry; ``format_diff`` prints them as a table;
``side_by_side_actograms`` draws the two actograms of one agent on one PNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from hamlet.config import ZONE_NAMES
from hamlet.metrics import routine, specialisation
from hamlet.metrics.common import DEFAULTS, after_burn_in

from sandbox import EXPLORATORY_TAG, assert_sandbox_output

N_HOUR_BINS = 8
STATE_COLUMNS = ("E", "F", "C")
NEED_NAMES = ("energy", "satiety", "social")
ZONE_COLOURS = {
    "HOME": "#4a5a8a", "FARM": "#7aa35c", "OFFICE": "#b08a3e", "CANTEEN": "#c9663a",
    "SOCIAL": "#a05a9a", "MARKET": "#3f8f8f", "TRANSIT": "#c8c4bb", "missing": "#ffffff",
}


def _source(df: pd.DataFrame, path: Optional[Path] = None) -> dict[str, Any]:
    first = df.iloc[0]
    return {"path": str(path) if path is not None else None, "condition": str(first["condition"]),
            "seed": int(first["seed"]), "checkpoint": str(first["checkpoint"]), **EXPLORATORY_TAG}


def fingerprint(df: pd.DataFrame, burn_in_days: Optional[int] = None, n_perm: int = 200,
                path: Optional[Path] = None) -> dict[str, Any]:
    """The fingerprint dictionary described in the module docstring; JSON-serialisable."""
    counts = specialisation.time_budget(df, burn_in_days=burn_in_days)                 # (N, 7)
    per_day = specialisation.time_budget(df, burn_in_days=burn_in_days, per_day=True)  # (N, days, 7)
    total = counts.sum()
    zone_fraction = counts.sum(axis=0) / total if total else np.zeros(len(ZONE_NAMES))
    row_totals = counts.sum(axis=1, keepdims=True)
    per_agent = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=np.float64), where=row_totals > 0)

    data = after_burn_in(df, burn_in_days)
    binned = routine.add_bins(data)
    hb = binned["hour_bin"].to_numpy().astype(np.int64)
    zone = binned["zone"].to_numpy().astype(np.int64)
    hour_zone = np.bincount(hb * len(ZONE_NAMES) + zone, minlength=N_HOUR_BINS * len(ZONE_NAMES))
    hour_zone = hour_zone.reshape(N_HOUR_BINS, len(ZONE_NAMES)).astype(np.float64)
    hour_zone = hour_zone / max(hour_zone.sum(), 1.0)

    states = {name: float(data[col].mean()) for name, col in zip(NEED_NAMES, STATE_COLUMNS)}
    per_need = {name: float(((1.0 - data[col].to_numpy(dtype=np.float64)) ** 2).mean())
                for name, col in zip(NEED_NAMES, STATE_COLUMNS)}
    dol = specialisation.dol_indiv(per_day, n_perm=n_perm, rng=np.random.default_rng(0))
    return {
        "zone_fraction": dict(zip(ZONE_NAMES, zone_fraction.tolist())),
        "zone_fraction_per_agent": per_agent.tolist(),
        "hour_zone": hour_zone.tolist(),
        "hour_zone_axes": {"rows": [f"{3 * k:02d}:00-{3 * k + 3:02d}:00" for k in range(N_HOUR_BINS)],
                           "columns": list(ZONE_NAMES)},
        "mean_state": states,
        "mean_drive": float(data["D"].mean()),
        "mean_drive_per_need": per_need,
        "dol_indiv": {"value": float(dol["value"]), "null_mean": float(dol["null_mean"]), "z": float(dol["z"])},
        "n_agents": int(counts.shape[0]),
        "days": int(per_day.shape[1]),
        "source": _source(df, path),
    }


def fingerprint_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Deltas ``b - a`` of every comparable entry of two fingerprints."""
    out: dict[str, Any] = {
        "zone_fraction": {z: b["zone_fraction"][z] - a["zone_fraction"][z] for z in ZONE_NAMES},
        "mean_state": {k: b["mean_state"][k] - a["mean_state"][k] for k in NEED_NAMES},
        "mean_drive": b["mean_drive"] - a["mean_drive"],
        "mean_drive_per_need": {k: b["mean_drive_per_need"][k] - a["mean_drive_per_need"][k] for k in NEED_NAMES},
        "dol_indiv": {k: b["dol_indiv"][k] - a["dol_indiv"][k] for k in ("value", "z")},
        "hour_zone": (np.asarray(b["hour_zone"]) - np.asarray(a["hour_zone"])).tolist(),
        "hour_zone_max_abs": float(np.abs(np.asarray(b["hour_zone"]) - np.asarray(a["hour_zone"])).max()),
        "sources": {"a": a["source"], "b": b["source"]},
        **EXPLORATORY_TAG,
    }
    return out


def format_diff(a: dict[str, Any], b: dict[str, Any], diff: Optional[dict[str, Any]] = None) -> str:
    """A fixed-width table of ``a``, ``b`` and their delta for every scalar entry."""
    diff = fingerprint_diff(a, b) if diff is None else diff
    rows = [("quantity", "a", "b", "b - a")]
    for z in ZONE_NAMES:
        rows.append((f"zone_fraction[{z}]", a["zone_fraction"][z], b["zone_fraction"][z], diff["zone_fraction"][z]))
    for k in NEED_NAMES:
        rows.append((f"mean_state[{k}]", a["mean_state"][k], b["mean_state"][k], diff["mean_state"][k]))
    rows.append(("mean_drive", a["mean_drive"], b["mean_drive"], diff["mean_drive"]))
    for k in NEED_NAMES:
        rows.append((f"drive_per_need[{k}]", a["mean_drive_per_need"][k], b["mean_drive_per_need"][k], diff["mean_drive_per_need"][k]))
    rows.append(("dol_indiv", a["dol_indiv"]["value"], b["dol_indiv"]["value"], diff["dol_indiv"]["value"]))
    rows.append(("dol_z", a["dol_indiv"]["z"], b["dol_indiv"]["z"], diff["dol_indiv"]["z"]))
    rows.append(("hour_zone max |delta|", "", "", diff["hour_zone_max_abs"]))
    width = max(len(r[0]) for r in rows)
    lines = []
    for r in rows:
        cells = [f"{r[0]:<{width}}"] + [f"{v:>9.3f}" if isinstance(v, float) else f"{v:>9}" for v in r[1:]]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _draw_actogram(ax, matrix: np.ndarray, title: str) -> None:
    from matplotlib.colors import ListedColormap

    names = list(ZONE_NAMES)
    cmap = ListedColormap([ZONE_COLOURS["missing"]] + [ZONE_COLOURS[n] for n in names])
    ax.imshow(matrix + 1, aspect="auto", cmap=cmap, vmin=0, vmax=len(names), interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("tick of day (6 game minutes)")
    ax.set_ylabel("day")
    ax.set_xticks([0, 60, 120, 180, 239])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])


def _legend(fig) -> None:
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=ZONE_COLOURS[n], label=n) for n in ZONE_NAMES], loc="lower center",
               ncol=len(ZONE_NAMES), fontsize=8, frameon=False)


def actogram_png(df: pd.DataFrame, agent: int, path: Path, ticks_per_day: Optional[int] = None,
                 title: Optional[str] = None) -> Path:
    """Write the actogram of ``agent`` to ``path`` (inside ``sandbox/runs/``) and return it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = assert_sandbox_output(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = routine.actogram_matrix(df, agent, ticks_per_day)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    _draw_actogram(ax, matrix, title or f"agent {agent}: {_source(df)['condition']} seed {_source(df)['seed']}")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def side_by_side_actograms(df_a: pd.DataFrame, df_b: pd.DataFrame, agent: int, path: Path,
                           labels: tuple[str, str] = ("a", "b"), ticks_per_day: Optional[int] = None) -> Path:
    """Two actograms of ``agent`` on one PNG, ``a`` above ``b``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = assert_sandbox_output(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, df, label in zip(axes, (df_a, df_b), labels):
        _draw_actogram(ax, routine.actogram_matrix(df, agent, ticks_per_day), f"{label}: agent {agent}")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def path_label(parquet: Path) -> str:
    """A file label that keeps a Parquet distinguishable from another run's file of the same name.

    The last three path parts without the suffix, joined by ``-``: the smoke
    run's ``smoke/seed0/eval/seed10000_ep0.parquet`` labels as
    ``seed0-eval-seed10000_ep0`` and a baseline
    ``A-S1-N8/GREEDY-CLOCK/seed10000_ep0.parquet`` as
    ``A-S1-N8-GREEDY-CLOCK-seed10000_ep0``, so two runs' files of the same
    evaluation seed never collide in one output directory.
    """
    parts = Path(parquet).with_suffix("").parts[-3:]
    return "-".join(parts)


def write_fingerprint(parquet: Path, out_dir: Path, agent: int = 0) -> tuple[Path, Path]:
    """``fingerprint`` and ``actogram_png`` for one file; returns ``(json_path, png_path)``."""
    out_dir = assert_sandbox_output(Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet, engine="pyarrow")
    fp = fingerprint(df, path=Path(parquet))
    label = path_label(parquet)
    json_path = out_dir / f"fingerprint_{label}.json"
    json_path.write_text(json.dumps(fp, indent=2))
    png_path = actogram_png(df, agent, out_dir / f"actogram_{label}_agent{agent}.png")
    return json_path, png_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path, help="one evaluation episode (LOG_COLUMNS)")
    parser.add_argument("--out", type=Path, required=True, help="output directory, inside sandbox/runs/")
    parser.add_argument("--agent", type=int, default=0, help="agent drawn in the actogram")
    parser.add_argument("--diff", type=Path, default=None, help="a second Parquet to compare against (deltas = second minus first)")
    args = parser.parse_args(argv)
    json_path, png_path = write_fingerprint(args.parquet, args.out, args.agent)
    fp_a = json.loads(json_path.read_text())
    print(f"wrote {json_path} and {png_path}")
    if args.diff is not None:
        json_b, _ = write_fingerprint(args.diff, args.out, args.agent)
        fp_b = json.loads(json_b.read_text())
        diff = fingerprint_diff(fp_a, fp_b)
        out_dir = assert_sandbox_output(args.out)
        la, lb = path_label(args.parquet), path_label(args.diff)
        (out_dir / f"diff_{la}_vs_{lb}.json").write_text(json.dumps(diff, indent=2))
        png = side_by_side_actograms(pd.read_parquet(args.parquet), pd.read_parquet(args.diff), args.agent,
                                     out_dir / f"actograms_{la}_vs_{lb}_agent{args.agent}.png",
                                     labels=(la, lb))
        print(format_diff(fp_a, fp_b, diff))
        print(f"wrote {png}")
    else:
        for zone, frac in fp_a["zone_fraction"].items():
            print(f"{zone:<8} {frac:6.3f}")
        print(f"mean drive {fp_a['mean_drive']:.3f}; DOL {fp_a['dol_indiv']['value']:.3f} (z {fp_a['dol_indiv']['z']:.2f})")


if __name__ == "__main__":
    main()
