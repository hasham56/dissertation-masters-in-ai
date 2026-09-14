#!/usr/bin/env bash
# Arm E2, exploratory: three seeds of A-S1 trained with solo social credit at half weight, then
# their evaluation and metrics. Nothing here is confirmatory: E2
# lives under its own root with its own settings hash and is reported without p-values.
#
#   bash scripts/run_e2.sh                       # runs/exploratory/E2, seeds 0 1 2
#   ROOT=runs/_x SEEDS="0" bash scripts/run_e2.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-runs}"
OUT="${E2_OUT:-$ROOT/exploratory/E2}"
SEEDS="${SEEDS:-0 1 2}"
THREADS="${THREADS:-2}"
N_EVAL="${N_EVAL:-32}"
JOBS="${JOBS:-4}"
SOLO="${SOLO:-0.5}"
mkdir -p "$OUT"

echo "== E2. training seeds $SEEDS at social_solo_fraction $SOLO into $OUT"
pids=()
for s in $SEEDS; do
    uv run python -m hamlet.train_fallback --arm A --symmetry S1 --seed "$s" \
        --out "$OUT" --threads "$THREADS" --social-solo-fraction "$SOLO" \
        >> "$OUT/launch.log" 2>&1 &
    pids+=($!)
    echo "   seed $s pid ${pids[-1]}"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
if [ "$rc" -ne 0 ]; then echo "a trainer exited $rc"; exit "$rc"; fi

echo "== E2. evaluation, stochastic pass on $N_EVAL seeds"
uv run python scripts/evaluate_grid.py --root "$OUT" --n-eval-seeds "$N_EVAL" --passes stochastic --jobs "$JOBS" || exit $?

echo "== E2. metrics, dashboard and division of labour only"
uv run python scripts/compute_metrics.py --root "$OUT" --n-eval-seeds "$N_EVAL" \
    --sections dol dashboard --no-baselines --jobs "$JOBS" || exit $?
echo "== E2. done"
