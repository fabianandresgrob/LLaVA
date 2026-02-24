#!/usr/bin/env python3
"""
Patch a DeepSpeed ZeRO-3 checkpoint so that the saved WarmupLR scheduler
state is compatible with HF Trainer's LambdaLR (cosine) scheduler.

The old WarmupLR state dict is missing the 'lr_lambdas' key that
LambdaLR.load_state_dict() requires, causing a KeyError on resume.
Adding 'lr_lambdas': None tells LambdaLR to keep its own lambda function
while still restoring last_epoch, so the cosine LR resumes from the
correct position in the schedule.

Usage:
    python scripts/fix_checkpoint_scheduler.py <checkpoint_dir>

Example:
    python scripts/fix_checkpoint_scheduler.py \
        $MCMLSCRATCH/checkpoints/llava-v1.5-7b-finetune-sae/checkpoint-3000
"""
import sys
import shutil
import pathlib
import torch


def find_state_file(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    latest_file = checkpoint_dir / "latest"
    if latest_file.exists():
        tag = latest_file.read_text().strip()
    else:
        candidates = sorted(checkpoint_dir.glob("global_step*"))
        if not candidates:
            raise FileNotFoundError(f"No global_step* directory found in {checkpoint_dir}")
        tag = candidates[-1].name
    state_file = checkpoint_dir / tag / "mp_rank_00_model_states.pt"
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")
    return state_file


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    checkpoint_dir = pathlib.Path(sys.argv[1])
    state_file = find_state_file(checkpoint_dir)

    print(f"Loading {state_file}  (may take a while for large checkpoints)...")
    state = torch.load(state_file, map_location="cpu")

    if "lr_scheduler" not in state:
        print("No lr_scheduler key in checkpoint — nothing to patch.")
        sys.exit(0)

    sched_state = state["lr_scheduler"]
    print(f"Current lr_scheduler state: {sched_state}")

    if "lr_lambdas" in sched_state:
        print("Already has lr_lambdas — no patch needed.")
        sys.exit(0)

    backup_file = state_file.with_suffix(".pt.bak")
    print(f"Backing up original to {backup_file}")
    shutil.copy2(state_file, backup_file)

    sched_state["lr_lambdas"] = None
    print(f"Patched lr_scheduler state: {sched_state}")

    print("Saving patched checkpoint...")
    torch.save(state, state_file)
    print("Done! You can now resume training normally.")


if __name__ == "__main__":
    main()
