#!/usr/bin/env python3
"""
Download SAE-finetuned LLaVA checkpoints, projectors, and SAE weights.
Run on the login node (compute nodes have no internet).

Usage:
    source sc_venv_template/activate.sh
    python scripts/download_sae_models.py --dry-run
    python scripts/download_sae_models.py
    python scripts/download_sae_models.py --sae-only
    python scripts/download_sae_models.py --models-only

After downloading, sae_checkpoint_path in each finetuned model's config.json is
patched to the local SAE path, since the path baked in at training time won't
exist on this cluster.

NOTE: The original LLaVA projectors are only needed if you re-run training.
For inference the projector weights are already embedded in the finetuned checkpoints.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# ── SAE weights ────────────────────────────────────────────────────────────────
# Set hf_repo if you uploaded ae.pt to HuggingFace, or local_src if the file
# already exists on this cluster (e.g. inside the sae-for-vlm repo).
SAES = {
    "imagenet": {
        "hf_repo":   "fabiangrob/imagenet-clip-l22-batchTopK-sae-k20-x8",   # e.g. "fabiangrob/sae-clip-imagenet"
        "local_src": "$SCRATCH/grob1/sae_checkpoints/imagenet/ae.pt",   # e.g. "/p/project1/taco-vlm/grob1/sae-for-vlm/checkpoints_dir/.../ae.pt"
    },
    "cc3m-laion": {
        "hf_repo":   "fabiangrob/cc3m-laion-clip-l22-batchTopK-sae-k20-x8",   # e.g. "fabiangrob/sae-clip-cc3m-laion"
        "local_src": "$SCRATCH/grob1/sae_checkpoints/cc3m-laion/ae.pt",   # e.g. "/p/project1/taco-vlm/grob1/sae-for-vlm/checkpoints_dir/.../ae.pt"
    },
}

# ── Original LLaVA pretrain projectors (public) ────────────────────────────────
# Used as the Stage 1 projector for the simple SAE (encode+decode) models.
# Only needed if you want to re-run training; the projector is already embedded
# in the finetuned checkpoints for inference.
ORIGINAL_PROJECTORS = [
    "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
    "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5",
]

# ── Simple SAE models (encode+decode) ─────────────────────────────────────────
# Uses the original LLaVA projector — no new pretrain stage.
# Verify these repo names match what you uploaded to HuggingFace.
SIMPLE_SAE_MODELS = [
    {
        "name":          "llava-7b-sae-imagenet",
        "finetune_repo": "fabiangrob/llava-v1.5-7b-finetune-sae",
        "sae":           "imagenet",
    },
    {
        "name":          "llava-7b-sae-cc3m-laion",
        "finetune_repo": "fabiangrob/llava-v1.5-7b-finetune-sae-cc3m-laion",
        "sae":           "cc3m-laion",
    },
    {
        "name":          "llava-13b-sae-imagenet",
        "finetune_repo": "fabiangrob/llava-v1.5-13b-finetune-sae",
        "sae":           "imagenet",
    },
    {
        "name":          "llava-13b-sae-cc3m-laion",
        "finetune_repo": "fabiangrob/llava-v1.5-13b-finetune-sae-cc3m-laion",
        "sae":           "cc3m-laion",
    },
]

# ── Encode-only SAE models ─────────────────────────────────────────────────────
# Requires a new Stage 1 pretrain because the projector input changes 1024→8192.
# pretrain_repo: the Stage 1 checkpoint (projector only, mm_projector.bin).
# finetune_repo: the full Stage 2 model — this is what you load for inference.
ENCODE_ONLY_MODELS = [
    {
        "name":          "llava-7b-encode-only-imagenet",
        "pretrain_repo": "fabiangrob/llava-v1.5-7b-pretrain-sae-encode-only",
        "finetune_repo": "fabiangrob/llava-v1.5-7b-finetune-sae-encode-only",
        "sae":           "imagenet",
    },
    {
        "name":          "llava-7b-encode-only-cc3m-laion",
        "pretrain_repo": "fabiangrob/llava-v1.5-7b-pretrain-sae-encode-only-cc3m-laion",
        "finetune_repo": "fabiangrob/llava-v1.5-7b-finetune-sae-encode-only-cc3m-laion",
        "sae":           "cc3m-laion",
    },
    {
        "name":          "llava-13b-encode-only-imagenet",
        "pretrain_repo": "fabiangrob/llava-v1.5-13b-pretrain-sae-encode-only",
        "finetune_repo": "fabiangrob/llava-v1.5-13b-finetune-sae-encode-only",
        "sae":           "imagenet",
    },
    {
        "name":          "llava-13b-encode-only-cc3m-laion",
        "pretrain_repo": "fabiangrob/llava-v1.5-13b-pretrain-sae-encode-only-cc3m-laion",
        "finetune_repo": "fabiangrob/llava-v1.5-13b-finetune-sae-encode-only-cc3m-laion",
        "sae":           "cc3m-laion",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def sae_dest(scratch: Path, sae_name: str) -> Path:
    return scratch / "grob1" / "checkpoints" / "sae" / sae_name / "ae.pt"


def check_env():
    missing = [v for v in ("HF_HOME", "SCRATCH") if not os.environ.get(v)]
    if missing:
        for v in missing:
            print(f"ERROR: ${v} is not set — add it to ~/.bashrc", file=sys.stderr)
        sys.exit(1)


def check_saes():
    errors = []
    for name, cfg in SAES.items():
        if cfg["hf_repo"] is None and cfg["local_src"] is None:
            errors.append(f"SAES['{name}']: set either hf_repo or local_src")
    if errors:
        print("ERROR: fill in SAE sources in this script before running:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def hf_download(repo_id: str, dry_run: bool) -> Path | None:
    from huggingface_hub import snapshot_download
    print(f"    {repo_id}")
    if dry_run:
        return None
    try:
        path = Path(snapshot_download(repo_id=repo_id, resume_download=True))
        print(f"    -> {path}")
        return path
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def fetch_sae(sae_name: str, cfg: dict, dest: Path, dry_run: bool) -> bool:
    if dest.exists():
        print(f"  Already present: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    if cfg["hf_repo"]:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading from HF: {cfg['hf_repo']}")
        if dry_run:
            return True
        try:
            hf_hub_download(repo_id=cfg["hf_repo"], filename="ae.pt",
                            local_dir=str(dest.parent))
            print(f"  -> {dest}")
            return True
        except Exception as e:
            print(f"  ERROR: {e}")
            return False
    else:
        src = Path(os.path.expandvars(cfg["local_src"]))
        print(f"  Copying from: {src}")
        if not src.exists():
            print(f"  ERROR: not found: {src}")
            return False
        if not dry_run:
            shutil.copy2(src, dest)
            print(f"  -> {dest}")
        return True


def patch_config(model_path: Path, new_sae_path: Path, dry_run: bool):
    config_file = model_path / "config.json"
    if not config_file.exists():
        return
    config = json.loads(config_file.read_text())
    old = config.get("sae_checkpoint_path")
    if old is None:
        return
    if old == str(new_sae_path):
        print(f"    config.json sae_checkpoint_path already correct")
        return
    print(f"    Patching sae_checkpoint_path:")
    print(f"      was: {old}")
    print(f"      now: {new_sae_path}")
    if not dry_run:
        config["sae_checkpoint_path"] = str(new_sae_path)
        config_file.write_text(json.dumps(config, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models-only", action="store_true", help="Skip SAE weights")
    parser.add_argument("--sae-only", action="store_true", help="Skip model downloads")
    parser.add_argument("--include-original-projectors", action="store_true",
                        help="Download original LLaVA pretrain projectors (only needed for re-training)")
    args = parser.parse_args()

    check_env()
    if not args.models_only:
        check_saes()

    scratch = Path(os.environ["SCRATCH"])
    print(f"HF_HOME : {os.environ['HF_HOME']}")
    print(f"SCRATCH : {scratch}")
    print()

    failed = []

    # ── SAE weights ─────────────────────────────────────────────────────────────
    if not args.models_only:
        print("=== SAE weights ===")
        for sae_name, cfg in SAES.items():
            dest = sae_dest(scratch, sae_name)
            print(f"[{sae_name}] -> {dest}")
            if not fetch_sae(sae_name, cfg, dest, args.dry_run):
                failed.append(f"SAE:{sae_name}")
        print()

    if not args.sae_only:
        # ── Simple SAE finetune models ───────────────────────────────────────────
        print("=== Simple SAE (encode+decode) — finetune checkpoints ===")
        for m in SIMPLE_SAE_MODELS:
            print(f"[{m['name']}]")
            local_path = hf_download(m["finetune_repo"], args.dry_run)
            if local_path is None and not args.dry_run:
                failed.append(m["finetune_repo"])
                continue
            if local_path:
                patch_config(local_path, sae_dest(scratch, m["sae"]), args.dry_run)
        print()

        # ── Encode-only models ───────────────────────────────────────────────────
        print("=== Encode-only SAE — finetune checkpoints (used for inference) ===")
        for m in ENCODE_ONLY_MODELS:
            print(f"[{m['name']}]")
            local_path = hf_download(m["finetune_repo"], args.dry_run)
            if local_path is None and not args.dry_run:
                failed.append(m["finetune_repo"])
                continue
            if local_path:
                patch_config(local_path, sae_dest(scratch, m["sae"]), args.dry_run)
        print()

        print("=== Encode-only SAE — pretrain projector checkpoints ===")
        for m in ENCODE_ONLY_MODELS:
            print(f"[{m['name']} projector]")
            local_path = hf_download(m["pretrain_repo"], args.dry_run)
            if local_path is None and not args.dry_run:
                failed.append(m["pretrain_repo"])
        print()

        # ── Original projectors (optional) ───────────────────────────────────────
        if args.include_original_projectors:
            print("=== Original LLaVA pretrain projectors ===")
            for repo in ORIGINAL_PROJECTORS:
                local_path = hf_download(repo, args.dry_run)
                if local_path is None and not args.dry_run:
                    failed.append(repo)
            print()

    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
        sys.exit(1)
    elif not args.dry_run:
        print("All downloads complete.")
        print("\nSAE paths (for submit_all.sh model entries):")
        for sae_name in SAES:
            print(f"  {sae_name}: {sae_dest(scratch, sae_name)}")


if __name__ == "__main__":
    main()
