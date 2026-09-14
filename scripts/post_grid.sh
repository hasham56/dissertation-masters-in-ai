#!/usr/bin/env bash
# Wait for the training grid, run the post-grid chain, and carry out the two-stage launch decision.
#
#   nohup bash scripts/post_grid.sh > /dev/null 2>&1 &          # arm it against runs/
#   ROOT=runs/_watch_test STAGE1=2 STAGE2=3 N_EVAL=2 POLL=5 bash scripts/post_grid.sh   # a test run
#
# Stage one: poll every POLL seconds until STAGE1 cells under ROOT hold selection.json and no
# trainer is alive, then run scripts/reproduce.sh once (evaluation passes, flags and the training
# table, metrics with their nulls, the manifest and energy totals, figures). Every step of that
# chain is resumable per run and per pass, so a second call over more cells only does the new work.
#
# Stage two: read ROOT/stage2_decision.txt, written before stage one by the pilot's social-level
# rule. If its first word is RUN, launch the remaining B-S1 and
# C-S1 cells with run_grid.py, wait for them the same way, and run reproduce.sh again so the new
# cells are evaluated and every aggregate table is rebuilt over all 48. If SKIP, say so and carry on
# to stage three; either way the grid is finished by this point.
#
# Stage three: the exploratory arms, in order.
#   E1        evaluation only, each A-S1 cell replayed with its last two agents scripted
#   E2        three seeds trained at social_solo_fraction 0.5, then their evaluation and metrics
# Each step writes the status file when it finishes, so a stall is visible at the step it stalled
# on. A step that fails stops the sequence: the ones after it read its outputs.
#
# NOTE: unlike its first version, this script does launch training, but only for stage two and only
# when the decision file says RUN. It never launches anything else, and it never edits the decision.
#
# Everything printed goes to ROOT/post_grid.log. ROOT/post_grid_status.txt is rewritten after each
# stage with that stage's result, so the morning's first question is answered by one file.
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-runs}"
STAGE1="${STAGE1:-32}"              # cells stage one is expected to produce (--grid minimum)
STAGE2="${STAGE2:-48}"              # cells after stage two (--grid confirmatory)
POLL="${POLL:-300}"                 # seconds between polls
MAX_WAIT="${MAX_WAIT:-86400}"       # give up waiting after this long (24 h)
N_EVAL="${N_EVAL:-32}"
JOBS="${JOBS:-4}"                   # safe once training is over, which is what we wait for
CONCURRENT="${CONCURRENT:-3}"       # stage two's launch concurrency
THREADS="${THREADS:-2}"
GRID_CMD="${GRID_CMD:-uv run python scripts/run_grid.py --grid confirmatory}"
# The chain itself, overridable so the two-stage orchestration can be tested with a stub instead of
# a real 20-minute run. It is invoked with ROOT, N_EVAL and JOBS exported.
CHAIN_CMD="${CHAIN_CMD:-bash scripts/reproduce.sh}"
# Stage three, each overridable so the sequence can be tested with stubs instead of hours of work.
E1_CMD="${E1_CMD:-uv run python scripts/exploratory_e1.py --root $ROOT --out $ROOT/exploratory/E1}"
E2_CMD="${E2_CMD:-bash scripts/run_e2.sh}"

LOG="$ROOT/post_grid.log"
STATUS="$ROOT/post_grid_status.txt"
DECISION="$ROOT/stage2_decision.txt"
mkdir -p "$ROOT"
exec >>"$LOG" 2>&1

# Trainers only. Matching this script's own command line would make it wait for itself;
# live_trainers below also requires the process to actually be a python or uv executable.
TRAINER_RE="${TRAINER_RE:-hamlet\.train_fallback|scripts/run_grid\.py|sandbox\.train}"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say()   { echo "[$(stamp)] $*"; }

# Only a process whose *executable* is python or uv counts as a trainer. A shell whose command
# line merely mentions one does not: a launcher wrapper, a terminal helper or an operator's own
# pgrep can carry the words for hours, and waiting for it to exit would stall the chain forever.
live_trainers() {
    local pid comm
    for pid in $(pgrep -f "$TRAINER_RE" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        comm=$(ps -o comm= -p "$pid" 2>/dev/null) || continue
        case "$comm" in
            python*|uv) ps -o pid=,args= -p "$pid" 2>/dev/null | cut -c1-140 ;;
        esac
    done
}
finished_cells() { find "$ROOT" -mindepth 3 -maxdepth 3 -name selection.json 2>/dev/null | wc -l; }

# wait_for_cells <n> <label>; returns 1 if it gave up.
wait_for_cells() {
    local want="$1" label="$2" waited=0 cells trainers
    while :; do
        cells=$(finished_cells)
        trainers=$(live_trainers)
        if [ "$cells" -ge "$want" ] && [ -z "$trainers" ]; then
            say "$label complete: $cells/$want cell(s) with selection.json, no trainer alive"
            return 0
        fi
        if [ "$waited" -ge "$MAX_WAIT" ]; then
            say "GAVE UP on $label after ${waited}s: $cells/$want cell(s), trainers: ${trainers:-none}"
            return 1
        fi
        say "waiting for $label: $cells/$want cell(s); trainers alive: $(echo "$trainers" | grep -c . || true)"
        sleep "$POLL"
        waited=$((waited + POLL))
    done
}

# run_chain <label>; sets CHAIN_RC and CHAIN_STAGES.
run_chain() {
    local label="$1" log_start
    say "running the chain for $label: $CHAIN_CMD (ROOT=$ROOT N_EVAL=$N_EVAL JOBS=$JOBS)"
    log_start=$(wc -l < "$LOG")
    ROOT="$ROOT" N_EVAL="$N_EVAL" JOBS="$JOBS" $CHAIN_CMD
    CHAIN_RC=$?
    say "the chain for $label exited $CHAIN_RC"
    CHAIN_STAGES=()
    while IFS= read -r line; do CHAIN_STAGES+=("$line"); done < <(
        tail -n +"$((log_start + 1))" "$LOG" | grep -oE '^== [0-9]+\. .*'
    )
}

# Stage three's steps, recorded so the status file says which ran and which is outstanding.
STEP_RESULTS=()

# run_step <label> <command string>; returns the command's exit code.
run_step() {
    local label="$1" cmd="$2" rc=0
    say "stage three, $label: $cmd"
    ROOT="$ROOT" N_EVAL="$N_EVAL" JOBS="$JOBS" THREADS="$THREADS" $cmd || rc=$?
    say "stage three, $label exited $rc"
    STEP_RESULTS+=("$([ "$rc" -eq 0 ] && echo PASS || echo FAIL)  $label")
    return "$rc"
}

# write_status <headline> [extra lines...]
write_status() {
    local headline="$1"; shift
    {
        echo "post-grid chain: $headline"
        echo "root        : $ROOT"
        echo "written     : $(stamp)"
        echo "log         : $LOG"
        for extra in "$@"; do echo "$extra"; done
        echo
        echo "stages entered by the last reproduce.sh run:"
        local last="" n=${#CHAIN_STAGES[@]}
        [ "$n" -gt 0 ] && last="${CHAIN_STAGES[$((n - 1))]}"
        if [ "$n" -eq 0 ]; then
            echo "  none reached"
        else
            for s in "${CHAIN_STAGES[@]}"; do
                [ -z "$s" ] && continue
                if [ "${CHAIN_RC:-1}" -ne 0 ] && [ "$s" = "$last" ]; then
                    echo "  FAIL  ${s#== }"
                else
                    echo "  PASS  ${s#== }"
                fi
            done
        fi
        if [ "${#STEP_RESULTS[@]}" -gt 0 ]; then
            echo
            echo "stage three:"
            for r in "${STEP_RESULTS[@]}"; do echo "  $r"; done
        fi
        echo
        echo "cells with selection.json: $(finished_cells)"
        echo "outputs:"
        for p in "$ROOT/metrics/per_seed.csv" "$ROOT/metrics/per_eval_seed.csv" "$ROOT/manifest.csv" \
                 "$ROOT/reports/grid_training.csv" "$ROOT/reports/grid_flags.csv" \
                 "$ROOT/reports/confirmatory.csv" "$ROOT/reports/results_tables.md" \
                 "$ROOT/exploratory/E1/e1.csv"; do
            if [ -f "$p" ]; then echo "  $(wc -l < "$p") line(s)  $p"; else echo "  MISSING     $p"; fi
        done
        echo "  $(find "$ROOT" -mindepth 3 -maxdepth 3 -name flags.json 2>/dev/null | wc -l) flags.json, "\
"$(find "$ROOT" -mindepth 4 -maxdepth 4 -name manifest.json -path '*/eval/*' 2>/dev/null | wc -l) eval manifest(s)"
        echo "  figures: ${FIGURES:-figures}/"
        if [ "${CHAIN_RC:-0}" -ne 0 ]; then
            echo
            echo "last 20 log lines:"
            tail -20 "$LOG" | sed 's/^/  /'
        fi
    } > "$STATUS"
    say "status written to $STATUS"
}

say "post_grid armed: root=$ROOT stage1=$STAGE1 stage2=$STAGE2 poll=${POLL}s n_eval=$N_EVAL jobs=$JOBS pid=$$"

CHAIN_RC=0
CHAIN_STAGES=()

# ---- stage one -------------------------------------------------------------------------------
if ! wait_for_cells "$STAGE1" "stage one"; then
    write_status "NOT RUN" "reason      : timed out waiting for $STAGE1 stage-one cell(s)"
    exit 1
fi
run_chain "stage one"
if [ "$CHAIN_RC" -ne 0 ]; then
    write_status "FAILED in stage one" "exit code   : $CHAIN_RC"
    exit "$CHAIN_RC"
fi
write_status "STAGE ONE PASSED" "stage two   : reading $DECISION"

# ---- stage two -------------------------------------------------------------------------------
# Stage two no longer ends the script: whether it runs or is skipped, stage three follows.
STAGE2_NOTE="not attempted"
if [ ! -f "$DECISION" ]; then
    say "no $DECISION; stage two not attempted"
    STAGE2_NOTE="not attempted ($DECISION does not exist)"
    write_status "STAGE ONE PASSED, STAGE TWO NOT ATTEMPTED" "reason      : $DECISION does not exist"
else
    verdict=$(grep -oE '\b(RUN|SKIP)\b' "$DECISION" | head -1 || true)
    say "stage-two decision: ${verdict:-unreadable}"
    if [ "$verdict" != "RUN" ]; then
        STAGE2_NOTE="skipped (decision ${verdict:-unreadable})"
        write_status "STAGE ONE PASSED, STAGE TWO SKIPPED" \
            "decision    : ${verdict:-unreadable} (from $DECISION)" \
            "note        : the 2x2 of B-S1 and C-S1 has no confirmatory test; the registered fallback applies"
    else
        say "stage two: launching $GRID_CMD --concurrent $CONCURRENT --threads $THREADS"
        $GRID_CMD --concurrent "$CONCURRENT" --threads "$THREADS"
        grid_rc=$?
        say "stage-two launcher exited $grid_rc"
        if [ "$grid_rc" -ne 0 ]; then
            write_status "STAGE TWO LAUNCH FAILED" "launcher rc : $grid_rc"
            exit "$grid_rc"
        fi
        if ! wait_for_cells "$STAGE2" "stage two"; then
            write_status "STAGE TWO INCOMPLETE" "reason      : timed out waiting for $STAGE2 cell(s)"
            exit 1
        fi
        run_chain "stage two"
        if [ "$CHAIN_RC" -ne 0 ]; then
            write_status "STAGE TWO CHAIN FAILED" "exit code   : $CHAIN_RC"
            exit "$CHAIN_RC"
        fi
        STAGE2_NOTE="ran ($STAGE2 cells)"
        write_status "BOTH GRID STAGES PASSED" "stage two   : $STAGE2_NOTE"
    fi
fi

# ---- stage three: the exploratory arms ----------------------------------
say "stage three: E1, then E2"
if ! run_step "E1 (evaluation only)" "$E1_CMD"; then
    write_status "STAGE THREE FAILED AT E1" "stage two   : $STAGE2_NOTE"
    exit 1
fi
write_status "STAGE THREE: E1 DONE" "stage two   : $STAGE2_NOTE"

if ! run_step "E2 (3 seeds, then evaluation and metrics)" "$E2_CMD"; then
    write_status "STAGE THREE FAILED AT E2" "stage two   : $STAGE2_NOTE"
    exit 1
fi
write_status "EVERYTHING DONE" "stage two   : $STAGE2_NOTE"
exit 0
