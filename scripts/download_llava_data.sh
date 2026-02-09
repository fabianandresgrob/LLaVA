#!/bin/bash
#
# Downloads all data needed for LLaVA 1.5 instruction tuning (Stage 2).
# Run this on the cluster, e.g.: sbatch scripts/download_llava_data_slurm.sh
# or directly: bash scripts/download_llava_data.sh
#
# Total size: ~95GB images + ~1GB annotations + ~50MB projector
#
# Required directory structure after download:
#   $DATA_DIR/
#   ├── llava_v1_5_mix665k.json
#   ├── coco/train2017/          (~19GB, 118K images)
#   ├── gqa/images/              (~20GB, 113K images)
#   ├── ocr_vqa/images/          (~33GB, needs dataset.json from Google Drive)
#   ├── textvqa/train_images/    (~7GB, 28K images)
#   └── vg/
#       ├── VG_100K/             (~15GB)
#       └── VG_100K_2/

set -e

# ---- Configure ----
if [ -z "$MCMLSCRATCH" ]; then
    echo "ERROR: \$MCMLSCRATCH is not set. Please set it to your scratch directory."
    exit 1
fi

DATA_DIR="${MCMLSCRATCH}/llava_data"
PROJECTOR_DIR="${MCMLSCRATCH}/checkpoints/llava-v1.5-7b-pretrain"

echo "=== LLaVA 1.5 Data Download ==="
echo "Data directory: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# ============================================================
# 1. Annotation JSON (~1GB)
# ============================================================
echo "[1/7] Downloading annotation JSON..."
if [ ! -f llava_v1_5_mix665k.json ]; then
    wget --progress=dot:giga \
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"
else
    echo "  Already exists, skipping."
fi

# ============================================================
# 2. COCO train2017 (~19GB)
# ============================================================
echo "[2/7] Downloading COCO train2017..."
if [ ! -d coco/train2017 ]; then
    mkdir -p coco
    wget --progress=dot:giga -O coco/train2017.zip \
        "http://images.cocodataset.org/zips/train2017.zip"
    cd coco && unzip -q train2017.zip && rm train2017.zip && cd ..
else
    echo "  Already exists, skipping."
fi

# ============================================================
# 3. GQA images (~20GB)
# ============================================================
echo "[3/7] Downloading GQA images..."
if [ ! -d gqa/images ]; then
    mkdir -p gqa
    wget --progress=dot:giga -O gqa/images.zip \
        "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    cd gqa && unzip -q images.zip && rm images.zip && cd ..
else
    echo "  Already exists, skipping."
fi

# ============================================================
# 4. TextVQA train images (~7GB)
# ============================================================
echo "[4/7] Downloading TextVQA images..."
if [ ! -d textvqa/train_images ]; then
    mkdir -p textvqa
    wget --progress=dot:giga -O textvqa/train_val_images.zip \
        "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
    cd textvqa && unzip -q train_val_images.zip && rm train_val_images.zip && cd ..
else
    echo "  Already exists, skipping."
fi

# ============================================================
# 5. VisualGenome part 1 (~15GB)
# ============================================================
echo "[5/7] Downloading VisualGenome part 1..."
if [ ! -d vg/VG_100K ]; then
    mkdir -p vg
    wget --progress=dot:giga -O vg/images.zip \
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
    cd vg && unzip -q images.zip && rm images.zip && cd ..
else
    echo "  Already exists, skipping."
fi

# ============================================================
# 6. VisualGenome part 2
# ============================================================
echo "[6/7] Downloading VisualGenome part 2..."
if [ ! -d vg/VG_100K_2 ]; then
    mkdir -p vg
    wget --progress=dot:giga -O vg/images2.zip \
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
    cd vg && unzip -q images2.zip && rm images2.zip && cd ..
else
    echo "  Already exists, skipping."
fi

# ============================================================
# 7. Stage 1 pretrained projector (~50MB)
# ============================================================
echo "[7/7] Downloading pretrained projector (Stage 1)..."
if [ ! -f "$PROJECTOR_DIR/mm_projector.bin" ]; then
    mkdir -p "$PROJECTOR_DIR"
    wget --progress=dot:mega -O "$PROJECTOR_DIR/mm_projector.bin" \
        "https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/resolve/main/mm_projector.bin"
else
    echo "  Already exists, skipping."
fi

# ============================================================
# OCR-VQA images (~33GB)
# ============================================================
# OCR-VQA dataset.json must be downloaded manually from Google Drive:
#   https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_
# Place it at: $DATA_DIR/ocr_vqa/dataset.json
# Then this script downloads the images from the URLs inside it.
echo ""
echo "[OCR-VQA] Downloading images from URLs in dataset.json..."
OCR_VQA_JSON="$DATA_DIR/ocr_vqa/dataset.json"
if [ ! -d "$DATA_DIR/ocr_vqa/images" ] || [ "$(ls "$DATA_DIR/ocr_vqa/images" 2>/dev/null | wc -l)" -lt 100 ]; then
    if [ -f "$OCR_VQA_JSON" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        # Don't let OCR-VQA failures kill the whole script (some URLs will be dead)
        set +e
        python "$SCRIPT_DIR/download_ocr_vqa.py" \
            --dataset_json "$OCR_VQA_JSON" \
            --output_dir "$DATA_DIR/ocr_vqa/images" \
            --workers 8
        set -e
    else
        echo ""
        echo "  WARNING: $OCR_VQA_JSON not found."
        echo "  Please download dataset.json from Google Drive:"
        echo "    https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_"
        echo "  Then place it at: $OCR_VQA_JSON"
        echo "  And re-run this script. (Other datasets are still downloaded.)"
        echo ""
    fi
else
    echo "  Already exists, skipping."
fi

# ============================================================
# Verify
# ============================================================
echo ""
echo "=== Download Summary ==="
echo "Data directory: $DATA_DIR"
echo ""
for d in coco/train2017 gqa/images ocr_vqa/images textvqa/train_images vg/VG_100K vg/VG_100K_2; do
    if [ -d "$DATA_DIR/$d" ]; then
        echo "  [OK] $d ($(du -sh "$DATA_DIR/$d" 2>/dev/null | cut -f1))"
    else
        echo "  [MISSING] $d"
    fi
done
echo ""
if [ -f "$DATA_DIR/llava_v1_5_mix665k.json" ]; then
    echo "  [OK] llava_v1_5_mix665k.json"
else
    echo "  [MISSING] llava_v1_5_mix665k.json"
fi
if [ -f "$PROJECTOR_DIR/mm_projector.bin" ]; then
    echo "  [OK] mm_projector.bin"
else
    echo "  [MISSING] mm_projector.bin"
fi
echo ""
echo "Done! You can now run training with:"
echo "  sbatch scripts/v1_5/slurm_finetune_sae.sh"
