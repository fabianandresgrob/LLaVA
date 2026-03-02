#!/bin/bash
# Tests signal handler logic with a realistic 5-process deepspeed hierarchy.
#
# Process tree (all match pgrep -f test_signal_deepspeed_sim.py):
#   runner (LAUNCHER_PID=$!) -> launcher (mimics launch.py) -> worker -> dataloader-0 -> dataloader-1
#
# Expected: SIGTERM reaches the worker (depth 2), checkpoint is saved.
# Failure:  SIGTERM hits launcher (depth 1) or a dataloader worker (depth 3).
#
# Run on the login node (no GPU needed):
#   bash scripts/v1_5/test_signal_3level.sh

set -e

CHECKPOINT_FILE="/tmp/test_checkpoint_saved.txt"
rm -f "$CHECKPOINT_FILE"

SCRIPT="scripts/v1_5/test_signal_deepspeed_sim.py"

ppid_of() { awk '/^PPid:/{print $2}' /proc/$1/status 2>/dev/null; }

handle_signal() {
    echo "$(date): SIGUSR1 received — finding worker PID..."
    PIDS=$(pgrep -f test_signal_deepspeed_sim.py | grep -v "^${LAUNCHER_PID}$")
    echo "$(date): Matching PIDs (excl. runner): $(echo $PIDS | tr '\n' ' ') | LAUNCHER_PID=$LAUNCHER_PID"

    LAUNCH_PID=""
    for pid in $PIDS; do
        if [ "$(ppid_of $pid)" = "$LAUNCHER_PID" ]; then
            LAUNCH_PID=$pid
            break
        fi
    done
    echo "$(date): launch.py PID: $LAUNCH_PID"

    WORKER_PID=""
    if [ -n "$LAUNCH_PID" ]; then
        for pid in $PIDS; do
            if [ "$(ppid_of $pid)" = "$LAUNCH_PID" ]; then
                WORKER_PID=$pid
                break
            fi
        done
    fi
    echo "$(date): Training worker PID: $WORKER_PID"

    if [ -n "$WORKER_PID" ]; then
        kill -TERM "$WORKER_PID" 2>/dev/null
    else
        echo "$(date): Worker PID not found! Falling back to launcher."
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
    fi
    while kill -0 "$LAUNCHER_PID" 2>/dev/null; do sleep 1; done
    echo "$(date): All processes exited."
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "✓ SUCCESS: Checkpoint was saved! (SIGTERM reached the worker)"
    else
        echo "✗ FAILURE: Checkpoint was NOT saved. (SIGTERM hit wrong process)"
    fi
    exit 0
}
trap handle_signal SIGUSR1

python "$SCRIPT" runner &
LAUNCHER_PID=$!
echo "Runner PID: $LAUNCHER_PID"
echo "Waiting 5s for process tree to start..."
sleep 5
echo "All sim PIDs after startup: $(pgrep -f test_signal_deepspeed_sim.py | tr '\n' ' ')"

# Simulate SLURM's early-warning signal after 15 seconds
(sleep 15 && kill -USR1 $$) &

wait "$LAUNCHER_PID" || true
echo "$(date): Done."
