#!/usr/bin/env bash
# The overnight sandbox campaign: four staged configs, sequential, exploratory only.
#
#   nohup bash sandbox/run_campaign.sh > sandbox/runs/campaign_summary/nohup.log 2>&1 &
#
# Launched by hand, and only when
# the study pilot is not using the machine: the script refuses to start while a study
# trainer is running. A failed config is recorded and the campaign CONTINUES to the
# next one; a broken lever must not cost the night. Everything is written under
# sandbox/runs/ and nothing under runs/ is touched.
#
# Per config, in order (neutral_anchor first, so the anchor exists for the diffs):
#   1. train seeds 0, 1, 2 at the staged 10M steps, evaluation rollouts included
#      (stdout in sandbox/runs/<name>/launch.log);
#   2. fingerprints and goals.json for every run;
#   3. for the three lever configs: seed-matched fingerprint diffs against
#      neutral_anchor (seed0 vs seed0, seed1 vs seed1, seed2 vs seed2, at each
#      evaluation seed), diff tables and side-by-side actograms into the run dir.
# Afterwards: the twelve training-return curves collected into
# sandbox/runs/campaign_summary/ (CSV + PNG) and a status table.
#
# SANDBOX_CAMPAIGN_TEST=1 runs the same pipeline at smoke scale (seed 0 only, two
# updates) into sandbox/runs/_campaign_test/ to verify the plumbing; it trains nothing
# worth keeping.

set -u
cd "$(dirname "$0")/.."     # repository root

CONFIGS=(neutral_anchor exploring_social setpoint_energy_07 collective_lambda_05)
ANCHOR=neutral_anchor
OUT="sandbox/runs"
SUMMARY="$OUT/campaign_summary"
TRAIN_ARGS=()
SEEDS_EXPECTED=(0 1 2)
EVAL_SEEDS=(10000 10001 10002)

if [ "${SANDBOX_CAMPAIGN_TEST:-0}" = "1" ]; then
    OUT="sandbox/runs/_campaign_test"
    SUMMARY="$OUT/campaign_summary"
    TRAIN_ARGS=(--seeds 0 --updates 2 --n-envs 2 --out "$OUT")
    SEEDS_EXPECTED=(0)
    echo "TEST MODE: smoke scale into $OUT"
fi

mkdir -p "$SUMMARY"
STATUS="$SUMMARY/status.txt"
: > "$STATUS"
note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

# Refuse to start beside the study pilot; the campaign yields, never competes.
if pgrep -af "hamlet.train_fallback|scripts/run_grid" | grep -v grep | grep -q .; then
    note "REFUSED: a study trainer is running; the pilot outranks the campaign."
    exit 1
fi

# Fail fast on any schema problem before burning the night.
for name in "${CONFIGS[@]}"; do
    if ! uv run python -m sandbox.train "sandbox/configs/$name.json" --dry-run > /dev/null 2> "$SUMMARY/dryrun_$name.err"; then
        note "ABORTED: $name fails --dry-run (see $SUMMARY/dryrun_$name.err); nothing was trained."
        exit 1
    fi
done
note "pre-flight: all ${#CONFIGS[@]} configs dry-run clean; git $(git describe --always --dirty --tags 2>/dev/null || echo unknown)"

for name in "${CONFIGS[@]}"; do
    mkdir -p "$OUT/$name"
    LOG="$OUT/$name/launch.log"
    note "start $name (log: $LOG)"
    if uv run python -m sandbox.train "sandbox/configs/$name.json" "${TRAIN_ARGS[@]}" > "$LOG" 2>&1; then
        note "done  $name"
    else
        note "FAILED $name (exit $?); continuing with the next config"
        continue
    fi

    for s in "${SEEDS_EXPECTED[@]}"; do
        RUN="$OUT/$name/seed$s"
        [ -d "$RUN/eval" ] || { note "WARN $name seed$s has no eval directory"; continue; }
        for k in "${EVAL_SEEDS[@]}"; do
            P="$RUN/eval/seed${k}_ep0.parquet"
            [ -f "$P" ] || continue
            if [ "$name" = "$ANCHOR" ]; then
                uv run python -m sandbox.fingerprint "$P" --out "$RUN" --agent 0 \
                    > "$RUN/fingerprint_seed${k}.txt" 2>&1 || note "WARN fingerprint failed: $P"
            else
                A="$OUT/$ANCHOR/seed$s/eval/seed${k}_ep0.parquet"      # seed-matched: seedS vs seedS
                if [ -f "$A" ]; then
                    uv run python -m sandbox.fingerprint "$A" --diff "$P" --out "$RUN" --agent 0 \
                        > "$RUN/diff_table_seed${s}_eval${k}.txt" 2>&1 || note "WARN diff failed: $P"
                else
                    uv run python -m sandbox.fingerprint "$P" --out "$RUN" --agent 0 \
                        > "$RUN/fingerprint_seed${k}.txt" 2>&1 || note "WARN fingerprint failed: $P (no anchor to diff against)"
                fi
            fi
        done
        uv run python -m sandbox.readout "$RUN/eval/seed${EVAL_SEEDS[0]}_ep0.parquet" --out "$RUN/goals.json" \
            > /dev/null 2>&1 || note "WARN readout failed: $RUN"
    done
done

# Collect the training-return curves of every finished run into one CSV and one PNG.
OUT="$OUT" SUMMARY="$SUMMARY" CONFIGS="${CONFIGS[*]}" uv run python - <<'PY' || note "WARN curve collection failed"
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = Path(os.environ["OUT"])
summary = Path(os.environ["SUMMARY"])
frames = []
for name in os.environ["CONFIGS"].split():
    for run in sorted(out.glob(f"{name}/seed*/progress.csv")):
        df = pd.read_csv(run)
        df.insert(0, "config", name)
        df.insert(1, "seed", int(run.parent.name.removeprefix("seed")))
        frames.append(df)
if frames:
    curves = pd.concat(frames, ignore_index=True)
    curves.to_csv(summary / "training_returns.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {"neutral_anchor": "#4a5a8a", "exploring_social": "#a05a9a",
               "setpoint_energy_07": "#b08a3e", "collective_lambda_05": "#3f8f8f"}
    for (name, seed), grp in curves.groupby(["config", "seed"]):
        ax.plot(grp["agent_steps"], grp["episode_return_mean"],
                color=colours.get(name, "#888888"), alpha=0.75,
                label=name if seed == grp["seed"].min() else None)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8, frameon=False)
    ax.set_xlabel("agent-steps")
    ax.set_ylabel("mean episode return (training; each config on its own reward scale)")
    ax.set_title("sandbox campaign: training-return curves (exploratory)")
    fig.tight_layout()
    fig.savefig(summary / "training_returns.png", dpi=120)
    rows = curves.groupby(["config", "seed"]).tail(1)[["config", "seed", "agent_steps", "episode_return_mean", "mean_D"]]
    (summary / "final_returns.json").write_text(json.dumps(
        {"exploratory": True, "rows": rows.to_dict("records")}, indent=2))
    print(f"collected {rows.shape[0]} run(s) into {summary}")
else:
    print("no progress.csv found; nothing collected")
PY

note "campaign finished; curves and status under $SUMMARY"
