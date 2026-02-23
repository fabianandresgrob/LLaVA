"""
Minimal fake launcher — mimics what deepspeed does:
starts the worker as a subprocess and waits for it.
"""
import subprocess, sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
worker = subprocess.Popen([sys.executable, os.path.join(script_dir, "test_signal_worker.py")])
sys.exit(worker.wait())
