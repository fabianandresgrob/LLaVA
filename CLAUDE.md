# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLaVA (Large Language and Vision Assistant) is a multimodal model that connects a CLIP vision encoder to a LLM (Vicuna/LLaMA/Mistral) via a learned MLP projector. Version 1.5 uses a two-layer MLP with GELU (mlp2x_gelu) as the vision-language connector.

## Installation

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install -e .                              # base install
pip install -e ".[train]"                     # adds deepspeed, ninja, wandb
pip install flash-attn --no-build-isolation   # optional, for flash attention
```

Pinned versions: torch 2.1.2, transformers 4.37.2, accelerate 0.21.0, deepspeed 0.12.6.

## Training

Two-stage training process, both use DeepSpeed:

**Stage 1 — Pretrain (feature alignment):** Trains only the MLP projector while freezing the vision encoder and LLM. Uses DeepSpeed ZeRO-2.
```bash
bash scripts/v1_5/pretrain.sh
```

**Stage 2 — Finetune (visual instruction tuning):** Full model finetuning with DeepSpeed ZeRO-3.
```bash
bash scripts/v1_5/finetune.sh        # full finetuning
bash scripts/v1_5/finetune_lora.sh   # LoRA variant (lower VRAM)
```

Training entry point: `llava/train/train_mem.py` (flash attention) or `llava/train/train_xformers.py` (xformers for V100).

Key rule: keep global batch size constant = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`.

DeepSpeed configs: `scripts/zero2.json` (pretrain), `scripts/zero3.json` (finetune), `scripts/zero3_offload.json` (CPU offload for low VRAM).

## Evaluation

Evaluation uses greedy decoding (no beam search). Pattern for all benchmarks:
```bash
# 1. Generate predictions
python -m llava.eval.model_vqa_loader --model-path <model> --question-file <q> --image-folder <imgs> --answers-file <out>
# 2. Run benchmark-specific evaluation
python llava/eval/eval_<benchmark>.py ...
```

Per-benchmark scripts live in `scripts/v1_5/eval/`. Supported benchmarks: VQAv2, GQA, VisWiz, ScienceQA, TextVQA, POPE, MME, MMBench, MMBench-CN, SEED-Bench, LLaVA-Bench, MM-Vet.

## Serving / Inference

```bash
# CLI inference (supports 4-bit/8-bit quantization)
python -m llava.serve.cli --model-path <model> --image-file <image>

# Gradio web demo (launch in order)
python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.gradio_web_server --controller http://localhost:10000
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path <model>
```

Programmatic usage:
```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=get_model_name_from_path(model_path))
```

## Architecture

Three-component design:

1. **Vision Encoder** (`llava/model/multimodal_encoder/clip_encoder.py`): Frozen CLIP ViT-L/14 @ 336px. Extracts features from a configurable layer (default: second-to-last). Feature selection modes: `patch` (excludes CLS) or `cls_patch`.

2. **Multimodal Projector** (`llava/model/multimodal_projector/builder.py`): Maps vision features to LLM embedding space. Types: `mlp2x_gelu` (default for v1.5), `linear`, `identity`.

3. **Language Model** (`llava/model/language_model/`): LLM backbone with multimodal fusion. Implementations for LLaMA (`llava_llama.py`), Mistral (`llava_mistral.py`), MPT (`llava_mpt.py`).

**Core fusion logic** is in `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal()` (`llava/model/llava_arch.py`): encodes images through vision tower + projector, replaces `<image>` tokens with image embeddings, and masks image tokens in the loss.

## Key Constants (`llava/constants.py`)

- `IGNORE_INDEX = -100` — label masking for loss computation
- `IMAGE_TOKEN_INDEX = -200` — special token ID for `<image>` placeholder
- `DEFAULT_IMAGE_TOKEN = "<image>"` — image placeholder in conversations

## Data Format

Training data is a JSON list of conversations:
```json
[{"id": "unique_id", "image": "relative/path.jpg", "conversations": [
    {"from": "human", "value": "<image>\nQuestion"},
    {"from": "gpt", "value": "Answer"}
]}]
```

Images are stored under `./playground/data/` organized by dataset (coco/train2017, gqa/images, ocr_vqa/images, textvqa/train_images, vg/VG_100K{,_2}).

## Key Implementation Details

- **Conversation templates** (`llava/conversation.py`): Multiple formats — `v1` (Vicuna), `llama_2`, `mpt`, `plain` (pretraining). The template controls system prompt, role names, and separators.
- **Modality-grouped sampling** (`llava/train/llava_trainer.py`): When `--group_by_modality_length True`, batches contain either all image or all text-only samples. Speeds up training ~25%.
- **LazySupervisedDataset** (`llava/train/train.py`): Lazy-loads images on demand; tokenizes conversations with special handling for `<image>` token replacement.
- **Image aspect ratio modes**: `pad` (expand to square with mean-color padding), `anyres` (divide into patches at multiple resolutions), `square` (standard resize/crop).
- **LoRA/QLoRA**: Enabled via `--lora_enable`, configured with `--lora_r`, `--lora_alpha`, `--bits` (4/8/16).
- **Custom optimizer** in `LLaVATrainer`: Supports separate learning rate for the projector via `--mm_projector_lr`.

## NeurIPS 2026 Plan

See `PLAN.md` for the current implementation plan. This repo handles Workstream 3 (SAE mitigation):
- **Exp 3.1 (priority):** Encode-only projector — feed 8192d sparse SAE activations directly to a new projector, skip SAE decoder
- **Exp 3.3 (conditional):** Encode/decode retrain from stage 1

Branch: `hmgu-training`. SAE bottleneck code is in `llava/model/sae_bottleneck.py`.

Related repos:
- `../sae-for-vlm/` — SAE checkpoints (BatchTopKSAE, CLIP layer 22, 8192 features)
- `../lmms-eval/` — benchmark evaluation of trained models
- `../vlm-mechanistic-analysis/` — mechanistic analysis (Workstream 2)

## Package Manager

Use `uv` as default. Dependencies in `pyproject.toml`. For server training with conda, install into the conda env with `uv pip install -e .` or `pip install -e .`.

## Commit Message Style

Use a single short imperative line. No bullet body, no co-author trailer. Example: `Add SAE inference support in builder.py`
