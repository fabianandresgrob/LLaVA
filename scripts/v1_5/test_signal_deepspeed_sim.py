"""
5-process deepspeed hierarchy simulator.

Mirrors the real deepspeed + PyTorch DataLoader setup:
  runner (= deepspeed entry point, LAUNCHER_PID=$!)
    -> launcher (= deepspeed launch.py)
      -> worker (= training process with HF Trainer SIGTERM handler)
        -> dataloader-0 (forked from worker, inherits /proc/PID/cmdline)
        -> dataloader-1 (forked from worker, inherits /proc/PID/cmdline)

All five processes match 'pgrep -f test_signal_deepspeed_sim.py',
exactly as all five real processes match 'pgrep -f train_mem.py'.
The correct target is the worker (depth 2), not the dataloader workers (depth 3).
"""
import subprocess, sys, signal, time, pathlib, multiprocessing

CHECKPOINT_FILE = pathlib.Path("/tmp/test_checkpoint_saved.txt")
mode = sys.argv[1] if len(sys.argv) > 1 else "runner"

if mode == "runner":
    # Mimics deepspeed runner.py: spawns launcher as a subprocess
    proc = subprocess.Popen([sys.executable, __file__, "launcher"])
    sys.exit(proc.wait())

elif mode == "launcher":
    # Mimics deepspeed launch.py: spawns worker as a subprocess.
    # Has its OWN SIGTERM handler that hard-kills children (like real launch.py sigkill_handler).
    proc = subprocess.Popen([sys.executable, __file__, "worker"])

    def launcher_sigterm(sig, frame):
        print(f"[LAUNCHER] SIGTERM received — hard-killing worker {proc.pid}", flush=True)
        proc.kill()
        proc.wait()
        sys.exit(1)

    signal.signal(signal.SIGTERM, launcher_sigterm)
    sys.exit(proc.wait())

elif mode == "worker":
    # Spawn 2 fake DataLoader workers via fork().
    # fork() inherits /proc/PID/cmdline, so they also match pgrep -f test_signal_deepspeed_sim.py,
    # just like real PyTorch DataLoader workers match pgrep -f train_mem.py.
    def fake_dataloader_worker():
        time.sleep(300)

    ctx = multiprocessing.get_context("fork")
    dl_workers = [ctx.Process(target=fake_dataloader_worker) for _ in range(2)]
    for w in dl_workers:
        w.start()

    def sigterm_handler(sig, frame):
        print("[WORKER] SIGTERM received — saving checkpoint...", flush=True)
        time.sleep(2)  # simulate checkpoint write latency
        CHECKPOINT_FILE.write_text("checkpoint saved!\n")
        print(f"[WORKER] Checkpoint saved to {CHECKPOINT_FILE}. Exiting.", flush=True)
        for w in dl_workers:
            w.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    print("[WORKER] Fake training loop started.", flush=True)
    for step in range(300):
        time.sleep(1)
        if step % 10 == 0:
            print(f"[WORKER] step {step}/300", flush=True)
