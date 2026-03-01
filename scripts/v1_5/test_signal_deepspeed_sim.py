"""
3-level deepspeed process hierarchy simulator.

Mirrors the real deepspeed setup:
  runner (= deepspeed entry point, LAUNCHER_PID=$!)
    -> launcher (= deepspeed launch.py, has train_mem.py in cmdline)
      -> worker (= actual training process with HF Trainer SIGTERM handler)

All three processes match 'pgrep -f test_signal_deepspeed_sim.py',
exactly as all three real deepspeed processes match 'pgrep -f train_mem.py'.
"""
import subprocess, sys, signal, time, pathlib

CHECKPOINT_FILE = pathlib.Path("/tmp/test_checkpoint_saved.txt")
mode = sys.argv[1] if len(sys.argv) > 1 else "runner"

if mode == "runner":
    # Mimics deepspeed runner.py: spawns launch.py as a subprocess
    proc = subprocess.Popen([sys.executable, __file__, "launcher"])
    sys.exit(proc.wait())

elif mode == "launcher":
    # Mimics deepspeed launch.py: spawns the training worker as a subprocess.
    # Has its OWN SIGTERM handler that hard-kills children (like real launch.py).
    import os

    proc = subprocess.Popen([sys.executable, __file__, "worker"])

    def launcher_sigterm(sig, frame):
        print(f"[LAUNCHER] SIGTERM received — hard-killing worker {proc.pid}", flush=True)
        proc.kill()
        proc.wait()
        sys.exit(1)

    signal.signal(signal.SIGTERM, launcher_sigterm)
    sys.exit(proc.wait())

elif mode == "worker":
    # Mimics HF Trainer: SIGTERM sets a flag, saves checkpoint, exits cleanly.
    def sigterm_handler(sig, frame):
        print("[WORKER] SIGTERM received — saving checkpoint...", flush=True)
        time.sleep(2)  # simulate checkpoint write latency
        CHECKPOINT_FILE.write_text("checkpoint saved!\n")
        print(f"[WORKER] Checkpoint saved to {CHECKPOINT_FILE}. Exiting.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    print("[WORKER] Fake training loop started.", flush=True)
    for step in range(300):
        time.sleep(1)
        if step % 10 == 0:
            print(f"[WORKER] step {step}/300", flush=True)
