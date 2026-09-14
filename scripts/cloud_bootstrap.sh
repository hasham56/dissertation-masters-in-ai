#!/usr/bin/env bash
# Prepare a fresh Ubuntu 22.04 VM to run the study, then measure what it can do.
#
#   bash cloud_bootstrap.sh <ref-or-tarball> [concurrency] [threads] [repo-url]
#
# Two sources, chosen by what the first argument is:
#
#   tarball   an existing .tgz/.tar.gz, as produced by `git archive -o /tmp/hamlet.tgz HEAD`.
#             Use this when the repository has no reachable remote, which is the case here.
#             Pass the commit's description alongside it, because a git archive carries no .git:
#               GIT_DESCRIBE="$(git describe --always --dirty --tags)" \
#               GIT_COMMIT="$(git rev-parse HEAD)" \
#               bash cloud_bootstrap.sh /tmp/hamlet.tgz 32 2
#
#   ref       a tag or commit to check out from CLOUD_REPO.
#               CLOUD_REPO=git@github.com:you/dissertation.git \
#               bash cloud_bootstrap.sh study-v1-final 32 2
#
# PROVENANCE, AND THE ONE REAL DIFFERENCE BETWEEN THE MODES.
# Every cell's manifest.json records `git describe` and the commit, read by hamlet/train_common.py
# from the working tree. A tarball has no .git, so that call returns "unknown" and every cell of a
# tarball run carries unknown provenance. The launcher does not refuse this: it refuses a tree whose
# describe ends in "-dirty", and "unknown" does not. So in tarball mode this script records the
# describe and commit you pass into runs/cloud_host.json, which cloud_fetch.sh prints beside the
# manifest rows. Provenance is then pinned at the run root rather than per cell. Clone mode records
# it per cell as usual and is preferable whenever a remote is reachable.
#
# Installs uv, puts the code in place, syncs the pinned environment, runs the study suite, then
# reports physical cores and a measured agent-steps per second at the requested concurrency. Trains
# nothing that is kept: the measurement writes to a scratch root and deletes it.
#
# Idempotent. Run it twice and the second run re-uses the environment, re-lays the source, re-runs
# the suite and re-measures. It never deletes an existing runs/ directory.
set -uo pipefail

REF="${1:-}"
CONCURRENT="${2:-16}"
THREADS="${3:-2}"
REPO="${CLOUD_REPO:-${4:-}}"
DIR="${CLOUD_DIR:-$HOME/Dissertation}"
MEASURE_S="${MEASURE_S:-60}"

if [ -z "$REF" ]; then
    echo "usage: bash cloud_bootstrap.sh <ref-or-tarball> [concurrency] [threads] [repo-url]" >&2
    echo "       tarball: GIT_DESCRIBE=... GIT_COMMIT=... bash cloud_bootstrap.sh /tmp/hamlet.tgz 32 2" >&2
    echo "       ref    : CLOUD_REPO=<url> bash cloud_bootstrap.sh <tag-or-commit> 32 2" >&2
    exit 2
fi

# A readable file is a tarball; anything else is a git ref.
MODE=ref
case "$REF" in
    *.tgz|*.tar.gz) [ -f "$REF" ] && MODE=tarball ;;
esac
if [ "$MODE" = ref ] && [ -f "$REF" ]; then
    echo "'$REF' is a file but not a .tgz or .tar.gz; rename it or pass a git ref" >&2
    exit 2
fi
if [ "$MODE" = ref ] && [ -z "$REPO" ] && [ ! -d "$DIR/.git" ]; then
    echo "no clone at $DIR and no repository url: set CLOUD_REPO or pass it as the fourth argument" >&2
    exit 2
fi

# An activated virtualenv in the calling shell makes uv warn on every command and could point the
# run at the wrong interpreter. uv manages its own .venv inside the clone, so drop the inherited one.
unset VIRTUAL_ENV

say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
step() { echo; echo "=== $* ==="; }

step "1. system packages"
# git and rsync are needed by this script and by cloud_fetch.sh; curl fetches uv. On a stock
# Ubuntu 22.04 image git and curl are present, so this is usually a no-op.
need=""
for p in git curl rsync; do command -v "$p" >/dev/null || need="$need $p"; done
if [ -n "$need" ]; then
    say "installing:$need"
    sudo apt-get update -qq && sudo apt-get install -y -qq $need
else
    say "git, curl and rsync already present"
fi

step "2. uv"
if command -v uv >/dev/null; then
    say "uv already installed: $(uv --version)"
else
    say "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1
fi
# The installer puts uv in ~/.local/bin, which a non-login shell may not have on PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "uv is installed but not on PATH" >&2; exit 1; }
say "$(uv --version)"

step "3. source at $REF ($MODE mode)"
if [ "$MODE" = tarball ]; then
    # Unpack beside any existing runs/ rather than over it: the code is replaced, results are not.
    mkdir -p "$DIR"
    say "unpacking $(basename "$REF") into $DIR"
    tar -xzf "$REF" -C "$DIR" || { echo "could not unpack $REF" >&2; exit 1; }
    [ -f "$DIR/hamlet/config.py" ] || { echo "$REF does not look like the repository (no hamlet/config.py)" >&2; exit 1; }
    cd "$DIR" || exit 1
    DESCRIBE="${GIT_DESCRIBE:-unknown}"
    COMMIT="${GIT_COMMIT:-unknown}"
    if [ "$DESCRIBE" = unknown ]; then
        say "WARNING: no GIT_DESCRIBE given. Every cell will record provenance 'unknown' and so"
        say "         will runs/cloud_host.json. Re-run with GIT_DESCRIBE and GIT_COMMIT set."
    fi
    say "at $DESCRIBE (${COMMIT:0:12}), from a tarball; cells will record provenance 'unknown'"
else
    if [ -d "$DIR/.git" ]; then
        say "re-using the clone at $DIR"
        git -C "$DIR" fetch --all --tags --quiet || exit 1
    else
        say "cloning $REPO into $DIR"
        git clone --quiet "$REPO" "$DIR" || exit 1
    fi
    git -C "$DIR" checkout --quiet "$REF" || { echo "no such tag or commit: $REF" >&2; exit 1; }
    cd "$DIR" || exit 1
    DESCRIBE=$(git describe --tags --always --dirty)
    COMMIT=$(git rev-parse HEAD)
    say "at $DESCRIBE (${COMMIT:0:12})"
fi

step "4. pinned environment"
uv sync --extra rl || exit 1
uv lock --check || { echo "uv.lock does not match pyproject.toml" >&2; exit 1; }
say "environment synced from uv.lock"

step "5. study suite"
suite_rc=0
uv run python -m pytest tests -q || suite_rc=$?
if [ "$suite_rc" -ne 0 ]; then
    echo "the study suite failed (exit $suite_rc); this VM is not fit to run the study" >&2
    exit "$suite_rc"
fi
say "study suite green"

step "6. machine"
CORES_PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
CORES_LOG=$(nproc)
MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
MODEL=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)
echo "  host          : $(hostname)"
echo "  cpu           : ${MODEL:-unknown}"
echo "  physical cores: $CORES_PHYS"
echo "  logical cores : $CORES_LOG"
echo "  memory        : ${MEM_GB} GB"
echo "  disk free     : $(df -h . | awk 'NR==2 {print $4}')"

step "7. measured rate, ${MEASURE_S}s at concurrency $CONCURRENT"
# Real trainers on a scratch root, killed after the window. Nothing here is kept: the point is the
# rate this machine sustains with this many trainers competing, which is the only number that
# predicts the grid's wall-clock. Energy logging is off so codecarbon does not skew a short run.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/hamlet-rate-XXXXXX")"
pids=()
for s in $(seq 0 $((CONCURRENT - 1))); do
    uv run python -m hamlet.train_fallback --arm A --symmetry S1 --seed "$s" \
        --out "$SCRATCH" --threads "$THREADS" --no-codecarbon > "$SCRATCH/trainer_$s.log" 2>&1 &
    pids+=($!)
done
say "${#pids[@]} trainer(s) started; measuring for ${MEASURE_S}s"
sleep "$MEASURE_S"
for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
sleep 3
for p in "${pids[@]}"; do kill -KILL "$p" 2>/dev/null; done

RATE=$(SCRATCH="$SCRATCH" uv run python - <<'PY'
import os, glob
import pandas as pd
rows = []
for f in glob.glob(os.path.join(os.environ["SCRATCH"], "*", "seed*", "progress.csv")):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if len(d) and d["elapsed_s"].max() > 0:
        rows.append((d["agent_steps"].max(), d["elapsed_s"].max()))
if not rows:
    print("0 0 0")
else:
    total = sum(a for a, _ in rows)
    span = max(e for _, e in rows)
    print(f"{total/span:.0f} {total/span/len(rows):.0f} {len(rows)}")
PY
)
rm -rf "$SCRATCH"
set -- $RATE
AGG="${1:-0}"; PER="${2:-0}"; N="${3:-0}"
if [ "$AGG" = "0" ]; then
    echo "  no trainer produced a progress row in ${MEASURE_S}s; raise MEASURE_S and retry" >&2
else
    echo "  trainers measured    : $N"
    echo "  aggregate            : ${AGG} agent-steps/s"
    echo "  per trainer          : ${PER} agent-steps/s"
    echo "  a 30M-step cell      : $(awk -v r="$PER" 'BEGIN{printf "%.1f", 30032640/r/60}') min"
    echo "  a 32-cell grid at $CONCURRENT: $(awk -v r="$PER" -v c="$CONCURRENT" \
        'BEGIN{w=int(32/c); if (32%c) w++; printf "%.2f", w*30032640/r/3600}') h"
fi

# Host provenance for cloud_fetch.sh. The trainer's own manifest records the rate and the energy
# but not which machine produced them, so record that here, beside the run root.
mkdir -p runs
cat > runs/cloud_host.json <<JSON
{
  "host": "$(hostname)",
  "cpu": "${MODEL:-unknown}",
  "physical_cores": $CORES_PHYS,
  "logical_cores": $CORES_LOG,
  "memory_gb": $MEM_GB,
  "source_mode": "$MODE",
  "ref": "$REF",
  "git_describe": "$DESCRIBE",
  "git_commit": "$COMMIT",
  "cells_record_provenance": $([ "$MODE" = tarball ] && echo '"unknown (tarball has no .git)"' || echo '"per cell, from the clone"'),
  "measured_concurrency": $CONCURRENT,
  "measured_threads": $THREADS,
  "measured_window_s": $MEASURE_S,
  "measured_aggregate_steps_per_s": $AGG,
  "measured_per_trainer_steps_per_s": $PER,
  "measured_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
say "wrote runs/cloud_host.json"

step "ready"
echo "  next: bash scripts/cloud_run.sh <grid> <concurrency> <threads>"
