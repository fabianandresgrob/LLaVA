#!/bin/bash
# Tests signal handler logic with a realistic 3-level deepspeed process hierarchy.
#
# Process tree (all match pgrep -f test_signal_deepspeed_sim.py):
#   runner (LAUNCHER_PID=$!) -> launcher (mimics launch.py) -> worker (has SIGTERM handler)
#
# Expected: SIGTERM reaches the worker, checkpoint is saved.
# Failure:  SIGTERM hits launcher, which hard-kills the worker -> no checkpoint.
#
# Run on the login node (no GPU needed):
#   bash scripts/v1_5/test_signal_3level.sh

set -e

CHECKPOINT_FILE="/tmp/test_checkpoint_saved.txt"
rm -f "$CHECKPOINT_FILE"

SCRIPT="scripts/v1_5/test_signal_deepspeed_sim.py"

handle_signal() {
    echo "$(date): SIGUSR1 received — finding worker PID..."
    echo "$(date): All sim PIDs: $(pgrep -f test_signal_deepspeed_sim.py | tr '\n' ' ') | LAUNCHER_PID=$LAUNCHER_PID"
    WORKER_PID=$(pgrep -f test_signal_deepspeed_sim.py | grep -v "^${LAUNCHER_PID}$" | sort -n | tail -1)
    echo "$(date): Targeting PID $WORKER_PID"
    if [ -n "$WORKER_PID" ]; then
        kill -TERM "$WORKER_PID" 2>/dev/null
    else
        echo "$(date): Worker PID not found, falling back to launcher"
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
    fi
    while kill -0 "$LAUNCHER_PID" 2>/dev/null; do sleep 1; done
    echo "$(date): All processes exited."
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "✓ SUCCESS: Checkpoint was saved! (SIGTERM reached the worker)"
    else
        echo "✗ FAILURE: Checkpoint was NOT saved. (SIGTERM hit the launcher instead)"
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
