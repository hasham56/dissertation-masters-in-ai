"""The H1a "no effect" threshold, derived by the registered rule rather than by hand.

The registered rule: the threshold is **a quarter of
the planted control's margin over the reference**, rounded **up** to the nearest 0.005 bits per tick.
One anchor, not two. Both inputs are means over the calibration seeds 10000, 10001 and 10002 at state
terciles, from ``scripts/report_clockmi.py --section gain`` (``runs/reports/gain_real.csv``), and the
margin is the planted series minus GREEDY-CLOCK's own gain on the same seeds.

The dropped clause was "or two thirds of GREEDY-CLOCK's own tercile gain, whichever is larger". It
was harmless while the reference was quiet, but at N = 9 with three canteen seats the reference's own
gain (0.090) overtook the plant's margin (0.046), so the larger-of-two rule set a bar that the
primary positive control could not itself clear. Anchoring on the plant's margin keeps the threshold
a fraction of a detectable effect, which is what the rule was for. It is stable across the change of
world: the same rule reads 0.015 from the archived N = 8 gains and 0.015 at N = 9.

Run::

    uv run python scripts/derive_h1a_threshold.py
    uv run python scripts/derive_h1a_threshold.py --gain-csv <archived study>/reports/gain_real.csv

Read-only: it changes no file. The value it prints is what belongs in
``analysis_constants.H1A_THRESHOLD`` and in the plan.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_constants as K  # noqa: E402

TERCILES = "3 quantile"
REFERENCE = "GREEDY-CLOCK"
PLANTED = "planted on GREEDY-CLOCK"
MARGIN_SHARE = 1 / 4
STEP = 0.005


def round_up(value: float, step: float = STEP) -> float:
    """Up to the nearest ``step``. Exact multiples stay put, which the rule intends."""
    return math.ceil(round(value / step, 9)) * step


def derive(gain_csv: Path) -> dict[str, object]:
    """The plant's margin over the reference, a quarter of it, and the rounded value the rule fixes."""
    d = pd.read_csv(gain_csv)
    d = d[d["state_bins"] == TERCILES]
    missing = {REFERENCE, PLANTED} - set(d["policy"])
    if missing:
        raise SystemExit(f"{gain_csv} has no tercile rows for {sorted(missing)}")

    ref = d.loc[d["policy"] == REFERENCE, "gain_bits"]
    plant = d.loc[d["policy"] == PLANTED, "gain_bits"]
    margin = plant.mean() - ref.mean()
    quarter = MARGIN_SHARE * margin
    return {
        "seeds": sorted(d.loc[d["policy"] == REFERENCE, "seed"]),
        "reference_mean": ref.mean(), "reference_seeds": list(ref),
        "planted_mean": plant.mean(), "planted_seeds": list(plant),
        "margin": margin, "a_quarter_of_the_margin": quarter, "threshold": round_up(quarter),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gain-csv", default="runs/reports/gain_real.csv",
                    help="output of report_clockmi.py --section gain")
    args = ap.parse_args(argv)

    r = derive(Path(args.gain_csv))
    print(f"source: {args.gain_csv}, state bins {TERCILES!r}, seeds {r['seeds']}\n")
    print(f"  {REFERENCE:24s} gain per seed {[round(v, 4) for v in r['reference_seeds']]}  mean {r['reference_mean']:.4f}")
    print(f"  {PLANTED:24s} gain per seed {[round(v, 4) for v in r['planted_seeds']]}  mean {r['planted_mean']:.4f}\n")
    print(f"  the plant's margin over the reference : {r['planted_mean']:.4f} - {r['reference_mean']:.4f} "
          f"= {r['margin']:.4f}")
    print(f"  a quarter of that margin              : {r['a_quarter_of_the_margin']:.4f}")
    print(f"  rounded up to the nearest {STEP}       : {r['threshold']:.3f} bits per tick\n")
    registered = K.H1A_THRESHOLD.value
    verdict = "unchanged" if abs(r["threshold"] - registered) < 1e-12 else f"MOVES from {registered:.3f}"
    print(f"registered H1A_THRESHOLD is {registered:.3f} (plan sections {K.H1A_THRESHOLD.section}): {verdict}")


if __name__ == "__main__":
    main()
