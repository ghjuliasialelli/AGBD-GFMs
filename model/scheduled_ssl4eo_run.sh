#!/bin/bash
#
# Deferred launcher for inference_ssl4eo.sh: sleep until a wall-clock time, wait for the GPU to be
# free, then run the three-tile job.
#
# Run detached so it survives logout:
#     setsid nohup bash model/scheduled_ssl4eo_run.sh 18:00 > /dev/null 2>&1 &
#
# A systemd --user timer was the obvious alternative but this account has Linger=no, so the user
# manager (and any timer it owns) is torn down when the last session ends. KillUserProcesses is the
# default `no`, so a setsid/nohup process outlives logout where the timer would not.
#
# Cancel with the PID in $PIDFILE -- NOT `pkill -f scheduled_ssl4eo_run`, which in this repo has
# previously matched and killed the calling shell.
#
set -u

TARGET="${1:-18:00}"
REPO=/scratch3/gsialelli/AGBD-GFMs
OUT_ROOT=/scratch3/gsialelli/ssl4eo_maps
LOG="$OUT_ROOT/logs/scheduled_run.log"
PIDFILE="$OUT_ROOT/logs/scheduled_run.pid"

# How long to keep waiting for a busy GPU before giving up rather than risking an OOM alongside
# whatever is using it. Concurrent loads have killed jobs on this box repeatedly.
GPU_WAIT_MAX_S=7200
GPU_POLL_S=300

mkdir -p "$OUT_ROOT/logs"
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG" ; }

# Seconds until the next occurrence of TARGET (today if still ahead, else tomorrow) ---------------
now=$(date +%s)
when=$(date -d "today $TARGET" +%s 2>/dev/null) || { log "FATAL: cannot parse time '$TARGET'"; exit 1; }
if [ "$when" -le "$now" ] ; then when=$(date -d "tomorrow $TARGET" +%s) ; fi
delay=$(( when - now ))

log "=================================================================="
log "Scheduled run armed (pid $$). Target $TARGET = $(date -d "@$when"), in $((delay / 60)) min."
log "Cancel with: kill $$   (do NOT use pkill -f)"

sleep "$delay"

log "Woke at target time. Checking the GPU is free..."

# Wait for a free GPU. Queried via nvidia-smi rather than pgrep: matching process names has
# deadlocked a wait loop in this repo before, when a wrapper shell's command line contained the
# pattern being matched.
waited=0
while : ; do
    # NOTE: do NOT write this as `... | grep -c . || echo 0`. `grep -c` prints 0 *and exits 1* when
    # it matches nothing, so the `|| echo 0` fires too and $busy becomes the two-line string "0\n0".
    # `[ "0\n0" -eq 0 ]` is then an `integer expression expected` error, which test reports as
    # false -- i.e. the guard reads a completely free GPU as permanently busy. That cost a whole
    # scheduled run on 2026-07-21: it waited out the full 2 h and aborted with the GPU idle.
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )
    case "$busy" in ''|*[!0-9]*) log "WARN: unparseable GPU process count '$busy'; assuming free." ; busy=0 ;; esac
    if [ "$busy" -eq 0 ] ; then
        log "GPU free. Launching."
        break
    fi
    if [ "$waited" -ge "$GPU_WAIT_MAX_S" ] ; then
        log "ABORT: GPU still busy after $((GPU_WAIT_MAX_S / 60)) min ($busy process(es)). Not"
        log "       starting -- a concurrent run risks OOM. Relaunch by hand when clear:"
        log "       bash $REPO/model/inference_ssl4eo.sh"
        rm -f "$PIDFILE"
        exit 1
    fi
    log "GPU busy ($busy process(es)); re-checking in $((GPU_POLL_S / 60)) min."
    sleep "$GPU_POLL_S"
    waited=$(( waited + GPU_POLL_S ))
done

log "--- inference_ssl4eo.sh starting ---"
cd "$REPO" || { log "FATAL: cannot cd to $REPO"; rm -f "$PIDFILE"; exit 1; }
bash model/inference_ssl4eo.sh >> "$LOG" 2>&1
rc=$?
log "--- inference_ssl4eo.sh finished, exit $rc ---"
if [ "$rc" -eq 137 ] ; then
    log "exit 137 = SIGKILL, almost certainly the OOM killer. Check free memory and rerun;"
    log "the job is resumable (skip-if-output-exists), so completed tiles will be skipped."
fi

rm -f "$PIDFILE"
