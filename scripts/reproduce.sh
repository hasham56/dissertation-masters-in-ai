#!/usr/bin/env bash
# Regenerate every post-grid table and figure from the Parquet logs, in order. Trains nothing.
#
#   bash scripts/reproduce.sh              # runs/ (the study)
#   ROOT=runs/_smoke_post N_EVAL=4 bash scripts/reproduce.sh   # the smoke cells
#
# Each step is resumable and skips work whose outputs exist; delete an output to redo it.
# Requires the pinned environment: uv sync --extra rl (uv.lock pins every package).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-runs}"
N_EVAL="${N_EVAL:-32}"
# The condition name carries the population size; ask the config rather than writing it out.
COND="$(uv run python -c 'from hamlet.config import HamletConfig; print(HamletConfig().condition_name)')"
JOBS="${JOBS:-1}"          # compute_metrics.py processes; the grid must be finished before using more than 1

echo "== 0. environment"
uv lock --check
uv run python -c "import hamlet, torch, codecarbon; print('hamlet ok, torch', torch.__version__, 'codecarbon', codecarbon.__version__)"

echo "== 1. baselines on the ${N_EVAL} evaluation seeds"
LAST_SEED=$((10000 + N_EVAL - 1))
if [ -f "$ROOT/$COND/GREEDY-CLOCK/seed${LAST_SEED}_ep0.parquet" ]; then
    echo "baseline Parquets present through seed ${LAST_SEED}; run_baselines.py not called (its summary CSV would be rewritten for ${N_EVAL} seeds)"
else
    uv run python scripts/run_baselines.py --n-seeds "$N_EVAL" --out "$ROOT"
fi
if [ -f "$ROOT/$COND/baseline_summary_n32.csv" ] && [ -f "$ROOT/$COND/baseline_summary.csv" ]; then
    uv run python scripts/report_grid.py --section baseline-gain --greedy-root "$ROOT"
fi

echo "== 2. evaluation passes on every finished run (stochastic, argmax, shocks, init-dispersion, swap)"
uv run python scripts/evaluate_grid.py --root "$ROOT" --n-eval-seeds "$N_EVAL" --passes all --jobs "$JOBS"

echo "== 3. training table and degeneracy flags"
uv run python scripts/report_grid.py --root "$ROOT" --greedy-root "$ROOT" --n-eval-seeds "$N_EVAL" --section all

echo "== 4. metrics with nulls (200 draws, per-section seeded streams) -> ${ROOT}/metrics/per_seed.csv"
uv run python scripts/compute_metrics.py --root "$ROOT" --n-eval-seeds "$N_EVAL" --passes all --jobs "$JOBS"

echo "== 5. reproducibility manifest and energy totals"
uv run python scripts/aggregate_manifest.py --root "$ROOT"

echo "== 6. confirmatory contrasts -> ${ROOT}/reports/confirmatory.{csv,md}"
uv run python scripts/confirmatory.py --root "$ROOT"

echo "== 7. figures and results tables -> ${ROOT}/figures/, ${ROOT}/reports/"
uv run python scripts/make_figures.py --root "$ROOT"
