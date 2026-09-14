#!/usr/bin/env bash
# Delete this VM once it has been idle for two hours. Runs from cron on the VM itself.
#
#   bash cloud_watchdog.sh --check          # print what it sees and exit, deleting nothing
#   bash cloud_watchdog.sh --install        # add the cron entry (every 5 minutes)
#   bash cloud_watchdog.sh --uninstall      # remove it
#   bash cloud_watchdog.sh                  # one pass, as cron runs it
#
# Flags
#   --idle-minutes N   minutes of idleness before deletion   (default 120)
#   --state PATH       where the last-busy stamp lives       (default ~/.hamlet_watchdog)
#   --check            report and exit 0, never delete
#   --dry-run          do everything except the delete call
#
# Why this exists. A grid once finished at 02:15 UTC and the machine then sat untouched until
# 09:37, because the "delete after two idle hours" rule lived only in an operator's head and no
# operator was awake. That gap cost about $8.30 and bought nothing. The rule belongs on the machine.
#
# What counts as busy, and the trap this avoids. Training and the post-grid chain are python or uv
# processes, so those are the primary signal. But a finished machine is still needed while its
# results are being pulled off it, and a pull is rsync and sshd -- no python anywhere. A watchdog
# that watched python alone would have deleted this VM in the middle of the 1.5-hour, 3.3 GB fetch
# that recovered the grid, destroying 32 cells to save a dollar. So an active transfer counts as
# busy too, and the machine is only idle when neither is running.
set -uo pipefail

IDLE_MINUTES=120
STATE="$HOME/.hamlet_watchdog"
MODE="run"
DRY=0
LOG="$HOME/hamlet_watchdog.log"

need() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
    case "$1" in
        --idle-minutes) need --idle-minutes $#; IDLE_MINUTES="$2"; shift 2 ;;
        --state)        need --state $#;        STATE="$2"; shift 2 ;;
        --check)        MODE="check"; shift ;;
        --install)      MODE="install"; shift ;;
        --uninstall)    MODE="uninstall"; shift ;;
        --dry-run)      DRY=1; shift ;;
        -h|--help)      sed -n '2,26p' "$0"; exit 0 ;;
        *)              echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$IDLE_MINUTES" in ''|*[!0-9]*) echo "--idle-minutes must be a whole number" >&2; exit 2 ;; esac
[ "$IDLE_MINUTES" -lt 1 ] && { echo "--idle-minutes must be at least 1" >&2; exit 2; }

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# ---- what the machine is doing ----------------------------------------------------------------
# A compute process: python or uv, and not this script or its own pgrep.
compute_procs() {
    local pid comm
    for pid in $(pgrep -f 'train_fallback|run_grid|compute_metrics|evaluate_grid|confirmatory|make_figures|reproduce\.sh|post_grid\.sh' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        comm=$(ps -o comm= -p "$pid" 2>/dev/null) || continue
        case "$comm" in
            python*|uv) ps -o pid=,args= -p "$pid" 2>/dev/null | cut -c1-120 ;;
        esac
    done
}

# A transfer: someone is pulling results off this machine right now.
transfer_procs() {
    local pid comm
    for pid in $(pgrep -x 'rsync|scp|sftp-server' 2>/dev/null); do
        comm=$(ps -o comm= -p "$pid" 2>/dev/null) || continue
        ps -o pid=,args= -p "$pid" 2>/dev/null | cut -c1-120
    done
}

N_COMPUTE=$(compute_procs | grep -c . || true)
N_TRANSFER=$(transfer_procs | grep -c . || true)
BUSY=$(( N_COMPUTE + N_TRANSFER ))

now=$(date +%s)
[ -f "$STATE" ] || echo "$now" > "$STATE"
if [ "$BUSY" -gt 0 ]; then
    echo "$now" > "$STATE"
fi
last=$(cat "$STATE" 2>/dev/null || echo "$now")
case "$last" in ''|*[!0-9]*) last="$now"; echo "$now" > "$STATE" ;; esac
idle_s=$(( now - last ))
idle_m=$(( idle_s / 60 ))

# ---- which instance is this --------------------------------------------------------------------
metadata() {
    curl -s --max-time 5 -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null
}
NAME=$(metadata name)
ZONE=$(metadata zone); ZONE="${ZONE##*/}"

report() {
    say "compute processes : $N_COMPUTE"
    say "transfers         : $N_TRANSFER"
    say "idle for          : ${idle_m} min of ${IDLE_MINUTES}"
    say "instance          : ${NAME:-<not on GCE>} ${ZONE:+in $ZONE}"
    if [ "$BUSY" -gt 0 ]; then
        compute_procs | sed 's/^/    compute  /'
        transfer_procs | sed 's/^/    transfer /'
    fi
}

case "$MODE" in
check)
    report
    if [ "$BUSY" -gt 0 ]; then say "verdict: BUSY, the clock is reset"
    elif [ "$idle_m" -ge "$IDLE_MINUTES" ]; then say "verdict: IDLE past the limit, this pass would DELETE"
    else say "verdict: idle, $(( IDLE_MINUTES - idle_m )) min to go"; fi
    exit 0 ;;
install)
    self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    line="*/5 * * * * bash $self --idle-minutes $IDLE_MINUTES >> $LOG 2>&1"
    (crontab -l 2>/dev/null | grep -v "cloud_watchdog.sh"; echo "$line") | crontab -
    say "installed: $line"
    exit 0 ;;
uninstall)
    (crontab -l 2>/dev/null | grep -v "cloud_watchdog.sh") | crontab -
    say "removed the watchdog cron entry"
    exit 0 ;;
esac

# ---- the one pass cron runs ---------------------------------------------------------------------
[ "$BUSY" -gt 0 ] && exit 0
[ "$idle_m" -lt "$IDLE_MINUTES" ] && exit 0

report
if [ -z "$NAME" ] || [ -z "$ZONE" ]; then
    say "refusing to delete: cannot read this instance's name and zone from the metadata server"
    exit 1
fi
command -v gcloud >/dev/null || { say "refusing to delete: no gcloud on PATH"; exit 1; }

say "idle ${idle_m} min with nothing running: deleting $NAME in $ZONE"
if [ "$DRY" -eq 1 ]; then
    say "dry run, not calling delete"
    exit 0
fi
gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
