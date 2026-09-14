#!/usr/bin/env bash
# Launch training on a prepared VM and arm the post-grid chain behind it.
#
#   bash cloud_run.sh --grid minimum --concurrent 32 --threads 2
#   bash cloud_run.sh --conditions A-S1 --seeds 0 1 2 --concurrent 3 --threads 2 --out runs/v2/pilot
#
# Flags
#   --grid NAME          a named grid (minimum, confirmatory, full, ...)
#   --conditions A B     an explicit subset instead of a grid
#   --seeds 0 1 2        an explicit seed subset
#   --concurrent N       trainers at once          (default 3)
#   --threads N          threads per trainer       (default 2)
#   --out PATH           run root                  (default runs)
#   --jobs N             chain width               (default: --concurrent)
#   --no-chain           launch training only, do not arm post_grid.sh
#   --no-watchdog        do not install the idle watchdog
#   --idle-minutes N     idle minutes before the watchdog deletes the VM (default 120)
#   --dry-run            print what would run and exit, launching nothing
#
# Starts run_grid.py under nohup with its usual refusals intact -- a dirty tree, a live trainer, a
# lock, a settings-hash mismatch on a finished cell -- because those are what keep a cloud run
# comparable with a local one. It never passes --force. It then arms post_grid.sh with JOBS set to
# the launch concurrency, so evaluation and metrics use the whole machine once training is over,
# and writes <out>/cloud_status.txt.
#
# Use --no-chain for a pilot that a human must read before anything else runs. That is the point of
# a pilot, and an armed chain would carry straight on into the full evaluation.
#
# Exits as soon as the launch is confirmed. The VM can then be left alone.
set -uo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

GRID=""; CONDITIONS=(); SEEDS=(); CONCURRENT=3; THREADS=2; OUT="runs"; JOBS=""; CHAIN=1; DRY=0
WATCHDOG=1; IDLE_MINUTES=120

# A flag that takes a value must be followed by one. Without this check, `--grid` as the final
# argument leaves $# unchanged when `shift 2` fails, and the loop below never terminates.
need() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }

while [ $# -gt 0 ]; do
    case "$1" in
        --grid)        need --grid $#;       GRID="$2"; shift 2 ;;
        --concurrent)  need --concurrent $#; CONCURRENT="$2"; shift 2 ;;
        --threads)     need --threads $#;    THREADS="$2"; shift 2 ;;
        --out)         need --out $#;        OUT="$2"; shift 2 ;;
        --jobs)        need --jobs $#;       JOBS="$2"; shift 2 ;;
        --no-chain)    CHAIN=0; shift ;;
        --no-watchdog) WATCHDOG=0; shift ;;
        --idle-minutes) need --idle-minutes $#; IDLE_MINUTES="$2"; shift 2 ;;
        --dry-run)     DRY=1; shift ;;
        --conditions)  shift; while [ $# -gt 0 ] && case "$1" in --*) false;; *) true;; esac; do CONDITIONS+=("$1"); shift; done ;;
        --seeds)       shift; while [ $# -gt 0 ] && case "$1" in --*) false;; *) true;; esac; do SEEDS+=("$1"); shift; done ;;
        -h|--help)     sed -n '2,28p' "$0"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for n in "$CONCURRENT" "$THREADS" ${JOBS:+"$JOBS"}; do
    case "$n" in ''|*[!0-9]*) echo "concurrency, threads and jobs must be whole numbers, got '$n'" >&2; exit 2;; esac
done
[ "$CONCURRENT" -lt 1 ] && { echo "--concurrent must be at least 1" >&2; exit 2; }
JOBS="${JOBS:-$CONCURRENT}"
if [ -z "$GRID" ] && [ "${#CONDITIONS[@]}" -eq 0 ]; then
    echo "give either --grid NAME or --conditions ..." >&2; exit 2
fi
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# ---- refuse early, readably, rather than let nohup swallow it ---------------------------------
HOSTFILE="runs/cloud_host.json"
if [ ! -f "$HOSTFILE" ]; then
    echo "no $HOSTFILE: run cloud_bootstrap.sh on this VM first" >&2
    exit 2
fi
CORES=$(uv run python -c "import json;print(json.load(open('$HOSTFILE'))['physical_cores'])" 2>/dev/null || echo 0)
if [ "$CORES" -gt 0 ] && [ $((CONCURRENT * THREADS)) -gt "$CORES" ]; then
    say "warning: $CONCURRENT x $THREADS = $((CONCURRENT * THREADS)) exceeds $CORES physical cores;"
    say "         the per-trainer rate will fall. Continuing: oversubscription is sometimes deliberate."
fi
PROV=$(uv run python -c "import json;d=json.load(open('$HOSTFILE'));print(d.get('git_describe','unknown'), d.get('source_mode','?'))" 2>/dev/null || echo "unknown ?")
say "source: $PROV"

# ---- assemble the launcher's arguments ----------------------------------------------------------
ARGS=(--concurrent "$CONCURRENT" --threads "$THREADS" --out "$OUT")
[ -n "$GRID" ] && ARGS+=(--grid "$GRID")
[ "${#CONDITIONS[@]}" -gt 0 ] && ARGS+=(--conditions "${CONDITIONS[@]}")
[ "${#SEEDS[@]}" -gt 0 ] && ARGS+=(--seeds "${SEEDS[@]}")
mkdir -p "$OUT"
STATUS="$OUT/cloud_status.txt"

say "dry run: uv run python scripts/run_grid.py ${ARGS[*]}"
if ! uv run python scripts/run_grid.py "${ARGS[@]}" --dry-run > /tmp/cloud_dryrun.txt 2>&1; then
    echo "the launcher refused; nothing was started:" >&2
    sed 's/^/  /' /tmp/cloud_dryrun.txt >&2
    exit 1
fi
PENDING=$(grep -c '^\[pending\]\|^run ' /tmp/cloud_dryrun.txt || true)
SKIPPED=$(grep -c '^skip ' /tmp/cloud_dryrun.txt || true)
say "$PENDING cell(s) to run, $SKIPPED already complete"
if [ "$DRY" -eq 1 ]; then
    echo
    sed 's/^/  /' /tmp/cloud_dryrun.txt
    echo
    echo "  --dry-run: nothing was launched"
    exit 0
fi
[ "$PENDING" -eq 0 ] && { echo "nothing to do; every requested cell is already complete" >&2; exit 0; }

# ---- training ------------------------------------------------------------------------------------
say "launching"
nohup uv run python scripts/run_grid.py "${ARGS[@]}" > "$OUT/grid_nohup.log" 2>&1 &
GRID_PID=$!
sleep 20
if ! kill -0 "$GRID_PID" 2>/dev/null && ! grep -q '^run ' "$OUT/grid_nohup.log" 2>/dev/null; then
    echo "the launcher exited immediately; nothing is running:" >&2
    sed 's/^/  /' "$OUT/grid_nohup.log" >&2
    exit 1
fi
say "grid launcher pid $GRID_PID"

# ---- the chain, at the machine's width -------------------------------------------------------------
CHAIN_PID="not armed (--no-chain)"
if [ "$CHAIN" -eq 1 ]; then
    say "arming post_grid.sh with JOBS=$JOBS"
    ROOT="$OUT" JOBS="$JOBS" CONCURRENT="$CONCURRENT" THREADS="$THREADS" \
        setsid nohup bash scripts/post_grid.sh > /dev/null 2>&1 < /dev/null &
    CHAIN_PID=$!
    sleep 5
else
    say "chain NOT armed: training only, as asked"
fi

# The idle watchdog. A grid that finishes at 02:15 with nobody awake used to leave the machine
# billing until someone noticed; one such run cost 7.4 idle hours. The rule now lives on the
# VM rather than in an operator's head. It counts an in-progress fetch as activity, so it cannot
# delete the machine out from under the rsync that collects the results.
WATCHDOG_STATE="not installed (--no-watchdog)"
if [ "$WATCHDOG" -eq 1 ]; then
    if bash scripts/cloud_watchdog.sh --install --idle-minutes "$IDLE_MINUTES" >/dev/null 2>&1; then
        WATCHDOG_STATE="armed, deletes this VM after $IDLE_MINUTES idle minutes"
    else
        WATCHDOG_STATE="FAILED to install -- delete this VM by hand when the run is done"
        say "warning: could not install the idle watchdog"
    fi
fi

cat > "$STATUS" <<TXT
cloud run: LAUNCHED
written     : $(date -u +%Y-%m-%dT%H:%M:%SZ)
host        : $(hostname)
source      : $PROV
target      : ${GRID:+grid $GRID}${CONDITIONS[*]:+conditions ${CONDITIONS[*]}}${SEEDS[*]:+ seeds ${SEEDS[*]}}
out         : $OUT
concurrency : $CONCURRENT trainer(s), $THREADS thread(s) each
chain jobs  : $JOBS
cells       : $PENDING to run, $SKIPPED already complete
grid pid    : $GRID_PID   (log $OUT/grid_nohup.log)
chain pid   : $CHAIN_PID  (log $OUT/post_grid.log, status $OUT/post_grid_status.txt)
watchdog    : $WATCHDOG_STATE

To watch:    tail -f $OUT/grid_nohup.log
             tail -f $OUT/post_grid.log
To fetch:    bash scripts/cloud_fetch.sh <user>@<vm-ip> --subdir $OUT   (on the local machine)
TXT
say "wrote $STATUS"
echo
sed 's/^/  /' "$STATUS"
