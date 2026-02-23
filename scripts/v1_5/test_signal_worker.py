"""
Fake training worker that mimics HF Trainer's SIGTERM handler.
SIGTERM → saves 'checkpoint' → exits 0.
"""
import signal, time, sys, pathlib

CHECKPOINT_FILE = pathlib.Path("/tmp/test_checkpoint_saved.txt")

def sigterm_handler(sig, frame):
    print("[WORKER] SIGTERM received — saving checkpoint...", flush=True)
    time.sleep(2)  # simulate checkpoint write time
    CHECKPOINT_FILE.write_text("checkpoint saved!\n")
    print(f"[WORKER] Checkpoint saved to {CHECKPOINT_FILE}. Exiting.", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

print("[WORKER] Fake training loop started.", flush=True)
for step in range(300):
    time.sleep(1)
    if step % 5 == 0:
        print(f"[WORKER] step {step}/300", flush=True)
