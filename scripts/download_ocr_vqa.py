"""
Download OCR-VQA images from URLs in dataset.json.
LLaVA expects all images saved as .jpg in ocr_vqa/images/.

Usage:
    python scripts/download_ocr_vqa.py --dataset_json /path/to/dataset.json --output_dir /path/to/ocr_vqa/images

You must first manually download dataset.json from:
    https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_
"""

import json
import os
import sys
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_image(key, url, output_dir):
    """Download a single image, saving as .jpg."""
    output_file = os.path.join(output_dir, f"{key}.jpg")
    if os.path.exists(output_file):
        return key, True, "exists"
    try:
        urllib.request.urlretrieve(url, output_file)
        return key, True, "ok"
    except Exception as e:
        return key, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True,
                        help="Path to OCR-VQA dataset.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for images (e.g. $MCMLSCRATCH/llava_data/ocr_vqa/images)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download threads")
    args = parser.parse_args()

    with open(args.dataset_json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(data)
    print(f"OCR-VQA: {total} images to download -> {args.output_dir}")

    success = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, k, data[k]["imageURL"], args.output_dir): k
            for k in data.keys()
        }
        for i, future in enumerate(as_completed(futures), 1):
            key, ok, msg = future.result()
            if ok:
                if msg == "exists":
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
            if i % 1000 == 0 or i == total:
                print(f"  [{i}/{total}] downloaded={success} skipped={skipped} failed={failed}")

    print(f"\nDone: {success} downloaded, {skipped} already existed, {failed} failed")
    if failed > 0:
        print("Some images failed — this is normal, some URLs may be dead.")
        print("LLaVA training will skip missing images gracefully.")


if __name__ == "__main__":
    main()
