#!/usr/bin/env bash
# Bring a cloud run home. Run this on the LOCAL machine, not on the VM.
#
#   bash cloud_fetch.sh <user@host> [remote-dir] [--dry-run]
#   bash scripts/cloud_fetch.sh hashie@34.105.22.9
#   bash scripts/cloud_fetch.sh hashie@34.105.22.9 ~/Dissertation --dry-run
#
# Copies the run root back into the local tree at the same paths: training cells, metrics/,
# reports/, figures/ and exploratory/.
#
# No existing local cell is ever overwritten. A cell is a <condition>/seed<n>/ directory holding
# metadata.json, and it is the unit that costs an hour of compute and carries its own provenance;
# silently merging a remote one over a local one would produce a directory whose checkpoints,
# progress and manifest came from two different machines. Every such collision is listed and
# skipped, and the script says so at the end rather than burying it.
#
# The aggregate outputs -- metrics/, reports/, figures/, manifest.csv -- are different: they are
# derived, cheap to rebuild, and meant to be replaced wholesale. Any local copy is moved aside into
# runs/_replaced_<timestamp>/ first, so nothing is lost and the replacement is visible.
set -uo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV

REMOTE=""; REMOTE_DIR="\$HOME/hamlet"; SUBDIR="runs"; DRY=""
need() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
    case "$1" in
        --remote-dir) need --remote-dir $#; REMOTE_DIR="$2"; shift 2 ;;
        --subdir)     need --subdir $#;     SUBDIR="${2%/}"; shift 2 ;;
        --dry-run)    DRY="--dry-run"; shift ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        --*)          echo "unknown argument: $1" >&2; exit 2 ;;
        *)            [ -z "$REMOTE" ] && REMOTE="$1" || REMOTE_DIR="$1"; shift ;;
    esac
done

if [ -z "$REMOTE" ]; then
    echo "usage: bash cloud_fetch.sh <user@host> [--remote-dir PATH] [--subdir runs/v2/pilot] [--dry-run]" >&2
    exit 2
fi
case "$SUBDIR" in runs|runs/*) ;; *) echo "--subdir must be runs or below it, got '$SUBDIR'" >&2; exit 2 ;; esac
ROOT="$SUBDIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

command -v rsync >/dev/null || { echo "rsync is not installed locally" >&2; exit 1; }
say "checking the remote"
if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" "test -d $REMOTE_DIR/$SUBDIR" 2>/dev/null; then
    echo "cannot reach $REMOTE, or $REMOTE_DIR/$SUBDIR does not exist there" >&2
    exit 1
fi
say "remote $REMOTE:$REMOTE_DIR/$SUBDIR  ->  local $ROOT"

# ---- which cells exist on each side ------------------------------------------------------------
say "listing remote cells"
ssh "$REMOTE" "cd $REMOTE_DIR/$SUBDIR 2>/dev/null && find . -mindepth 3 -maxdepth 3 -name metadata.json -printf '%h\n' | sed 's#^\./##' | sort" \
    > /tmp/cloud_remote_cells.txt 2>/dev/null
REMOTE_N=$(grep -c . /tmp/cloud_remote_cells.txt || true)
find "$ROOT" -mindepth 3 -maxdepth 3 -name metadata.json -printf '%h\n' 2>/dev/null \
    | sed "s#^$ROOT/##" | sort > /tmp/cloud_local_cells.txt
LOCAL_N=$(grep -c . /tmp/cloud_local_cells.txt || true)
comm -12 /tmp/cloud_remote_cells.txt /tmp/cloud_local_cells.txt > /tmp/cloud_collisions.txt
comm -23 /tmp/cloud_remote_cells.txt /tmp/cloud_local_cells.txt > /tmp/cloud_new.txt
COLLIDE=$(grep -c . /tmp/cloud_collisions.txt || true)
NEW=$(grep -c . /tmp/cloud_new.txt || true)

echo
echo "  remote cells        : $REMOTE_N"
echo "  local cells         : $LOCAL_N"
echo "  new, will be copied : $NEW"
echo "  already here, SKIPPED: $COLLIDE"
if [ "$COLLIDE" -gt 0 ]; then
    sed 's/^/      /' /tmp/cloud_collisions.txt | head -40
    [ "$COLLIDE" -gt 40 ] && echo "      ... and $((COLLIDE - 40)) more"
fi

# Every colliding cell becomes an rsync exclusion, so the protection is enforced by rsync itself
# rather than by a later tidy-up.
: > /tmp/cloud_excludes.txt
while IFS= read -r c; do [ -n "$c" ] && echo "/$c/" >> /tmp/cloud_excludes.txt; done < /tmp/cloud_collisions.txt

# ---- move aside the derived outputs we are about to replace -------------------------------------
if [ -z "$DRY" ]; then
    BACKUP="$ROOT/_replaced_$STAMP"
    moved=0
    for p in metrics reports figures manifest.csv; do
        if [ -e "$ROOT/$p" ]; then
            mkdir -p "$BACKUP"
            mv "$ROOT/$p" "$BACKUP/" && moved=$((moved + 1))
        fi
    done
    [ "$moved" -gt 0 ] && say "moved $moved existing derived output(s) to $BACKUP"
fi

# ---- the transfer --------------------------------------------------------------------------------
say "copying${DRY:+ (dry run)}"
rsync -az --info=stats1,progress2 --partial $DRY \
      --exclude-from=/tmp/cloud_excludes.txt \
      --exclude='_smoke*' --exclude='dev/' --exclude='_study_n8/' \
      --exclude='*.pyc' --exclude='__pycache__/' \
      "$REMOTE:$REMOTE_DIR/$SUBDIR/" "$ROOT/"
rc=$?
[ "$rc" -ne 0 ] && { echo "rsync exited $rc" >&2; exit "$rc"; }
say "transfer complete"
[ -n "$DRY" ] && { echo; echo "  dry run: nothing was written"; exit 0; }

# ---- provenance ------------------------------------------------------------------------------------
echo
echo "=== provenance of what came back ==="
uv run python scripts/aggregate_manifest.py --root "$ROOT" --out "$ROOT/manifest.csv" > /dev/null 2>&1 \
    || say "note: could not rebuild manifest.csv; showing whatever came across"

ROOT="$ROOT" uv run python - <<'PY'
import json, os
from pathlib import Path
import pandas as pd

root = Path(os.environ["ROOT"])
host = root / "cloud_host.json"
if host.exists():
    h = json.loads(host.read_text())
    print("machine that produced these runs")
    for k in ("host", "cpu", "physical_cores", "logical_cores", "memory_gb", "ref",
              "git_describe", "measured_concurrency", "measured_threads",
              "measured_aggregate_steps_per_s", "measured_per_trainer_steps_per_s", "measured_utc"):
        if k in h:
            print(f"  {k:<34} {h[k]}")
else:
    print("  no cloud_host.json came back; host and core count are unknown")

m = root / "manifest.csv"
if not m.exists():
    print("\n  no manifest.csv; nothing to show per cell")
    raise SystemExit
d = pd.read_csv(m)
cols = [c for c in ("condition", "seed", "git_describe", "wall_clock_min",
                    "agent_steps_per_s", "energy_kwh", "emissions_kg", "status") if c in d]
print(f"\nmanifest rows: {len(d)}")
with pd.option_context("display.width", 200, "display.max_rows", 60, "display.precision", 3):
    print(d[cols].to_string(index=False))
num = [c for c in ("wall_clock_min", "agent_steps_per_s", "energy_kwh", "emissions_kg") if c in d]
if num:
    print("\ntotals and means")
    for c in num:
        v = pd.to_numeric(d[c], errors="coerce").dropna()
        if len(v):
            print(f"  {c:<20} total {v.sum():>12,.3f}   mean {v.mean():>10,.3f}")
PY

echo
echo "  cells skipped because they already existed here: $(grep -c . /tmp/cloud_collisions.txt || echo 0)"
echo "  (nothing local was overwritten; see the list above if that number is not zero)"
