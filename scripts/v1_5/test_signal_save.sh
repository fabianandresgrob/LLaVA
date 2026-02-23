#!/bin/bash
# Tests the signal handler logic from slurm_finetune_sae_chained.sh without a GPU.
# Run inside an interactive session:
#   srun --cpus-per-task=2 --mem=2G --time=00:10:00 --pty bash
#   bash scripts/v1_5/test_signal_save.sh

set -e

CHECKPOINT_FILE="/tmp/test_checkpoint_saved.txt"
rm -f "$CHECKPOINT_FILE"

# --- Same handler as slurm_finetune_sae_chained.sh ---
handle_signal() {
    echo "$(date): SIGUSR1 received — finding worker PID..."
    # LAUNCHER_PID already set via $! in outer scope — don't use jobs -p here
    # (the timer subshell may still appear in the job list, giving multiple PIDs)
    WORKER_PID=$(pgrep -f test_signal_worker.py | head -1)
    if [ -n "$WORKER_PID" ]; then
        echo "$(date): Found worker PID $WORKER_PID — sending SIGTERM"
        kill -TERM "$WORKER_PID" 2>/dev/null
    else
        echo "$(date): Worker PID not found! Falling back to killing launcher."
        kill -TERM "$LAUNCHER_PID" 2>/dev/null
    fi
    # 'wait' inside a trap handler can return early in bash — poll instead
    while kill -0 "$LAUNCHER_PID" 2>/dev/null; do sleep 1; done
    echo "$(date): Worker exited."
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "✓ SUCCESS: Checkpoint was saved!"
    else
        echo "✗ FAILURE: Checkpoint was NOT saved."
    fi
    exit 0
}
trap handle_signal SIGUSR1

# Start fake launcher (mimics deepspeed), which starts the worker as its child
python scripts/v1_5/test_signal_launcher.py &
LAUNCHER_PID=$!
echo "Launcher PID: $LAUNCHER_PID"

# Simulate SLURM's early-warning signal after 15 seconds
(sleep 15 && kill -USR1 $$) &

wait "$LAUNCHER_PID" || true
echo "$(date): Done."
