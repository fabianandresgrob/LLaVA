"""
Quick dataset health check: samples random images from each dataset
subfolder and verifies they can be opened by PIL.

Usage:
    python scripts/check_dataset_images.py --data_dir $MCMLSCRATCH/llava_data --samples 10
"""

import argparse
import os
import random
from pathlib import Path
from PIL import Image

DATASET_FOLDERS = [
    "coco/train2017",
    "gqa/images",
    "ocr_vqa/images",
    "textvqa/train_images",
    "vg/VG_100K",
    "vg/VG_100K_2",
]


def check_folder(data_dir: str, subfolder: str, num_samples: int):
    folder = os.path.join(data_dir, subfolder)
    if not os.path.isdir(folder):
        print(f"  MISSING  {subfolder}/")
        return 0, 0, 0

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
    total = len(files)
    if total == 0:
        print(f"  EMPTY    {subfolder}/ (no image files)")
        return 0, 0, 0

    samples = random.sample(files, min(num_samples, total))
    ok, bad = 0, 0
    bad_files = []
    for f in samples:
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.verify()
            ok += 1
        except Exception as e:
            bad += 1
            bad_files.append((f, str(e)))

    status = "OK" if bad == 0 else "CORRUPT"
    print(f"  {status:8s} {subfolder}/ — {total} files, sampled {len(samples)}, {ok} ok, {bad} bad")
    for f, err in bad_files:
        print(f"           BAD: {f} ({err})")
    return total, ok, bad


def main():
    parser = argparse.ArgumentParser(description="Check LLaVA dataset images")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to llava_data root")
    parser.add_argument("--samples", type=int, default=10, help="Number of random images to check per folder")
    parser.add_argument("--check_all", action="store_true", help="Check ALL images (slow)")
    args = parser.parse_args()

    print(f"\nDataset image health check: {args.data_dir}")
    print(f"Sampling {args.samples} random images per folder\n")

    total_files, total_ok, total_bad = 0, 0, 0
    for subfolder in DATASET_FOLDERS:
        n_samples = 0 if args.check_all else args.samples
        if args.check_all:
            folder = os.path.join(args.data_dir, subfolder)
            if os.path.isdir(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
                n_samples = len(files)
            else:
                n_samples = 0
        else:
            n_samples = args.samples
        t, o, b = check_folder(args.data_dir, subfolder, n_samples)
        total_files += t
        total_ok += o
        total_bad += b

    print(f"\nTotal: {total_files} image files across {len(DATASET_FOLDERS)} folders")
    if total_bad > 0:
        print(f"Found {total_bad} corrupt images!")
    else:
        print("All sampled images OK")


if __name__ == "__main__":
    main()
