#!/usr/bin/env python3
"""
Standalone diagnostic for the 'tokenization mismatch: 1 vs. N' bug.

Run on HMGU (no GPU needed):
    cd /ictstr01/home/eml/fabian.grob/projects/LLaVA
    conda activate llava
    # Synthetic test only:
    python scripts/debug_preprocess_v1.py
    # Test with real training data:
    python scripts/debug_preprocess_v1.py --data_path $SCRATCH/llava_data/llava_v1_5_mix665k.json

Checks whether preprocess_v1 correctly masks labels for image-containing samples.
"""

import sys, os, argparse

# Make sure llava is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from llava import conversation as conversation_lib
from llava.train.train import preprocess_v1, IS_TOKENIZER_GREATER_THAN_0_14
from llava.mm_utils import tokenizer_image_token
from llava.constants import IGNORE_INDEX
import tokenizers

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None,
                    help="Path to llava_v1_5_mix665k.json for real-data test")
parser.add_argument("--n_samples", type=int, default=20,
                    help="Number of image samples to test from real data")
args = parser.parse_args()

print("=" * 60)
print("Environment info:")
print(f"  transformers: {transformers.__version__}")
print(f"  tokenizers:   {tokenizers.__version__}")
print(f"  IS_TOKENIZER_GREATER_THAN_0_14: {IS_TOKENIZER_GREATER_THAN_0_14}")
print()

# Load tokenizer (same as in training)
MODEL = "lmsys/vicuna-7b-v1.5"
print(f"Loading tokenizer from {MODEL} ...")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL, use_fast=False, model_max_length=2048
)
tokenizer.pad_token = tokenizer.unk_token
print(f"  pad_token_id: {tokenizer.pad_token_id}")
print(f"  bos_token_id: {tokenizer.bos_token_id}")
print(f"  eos_token_id: {tokenizer.eos_token_id}")
print(f"  tokenizer.legacy: {getattr(tokenizer, 'legacy', 'not set')}")
print()

# Set conversation template (same as --version v1)
conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
conv = conversation_lib.default_conversation
print("Conversation template:")
print(f"  version:   {conv.version}")
print(f"  sep:       {repr(conv.sep)}")
print(f"  sep2:      {repr(conv.sep2)}")
print(f"  roles:     {conv.roles}")
print(f"  sep_style: {conv.sep_style}")
sep = conv.sep + conv.roles[1] + ": "
print(f"  derived sep for masking: {repr(sep)}")
print()

# ---------------------------------------------------------------
# Helper: run preprocess_v1 and return (n_valid, n_total, prompt)
# ---------------------------------------------------------------
def check_sources(sources, has_image):
    # Build the prompt string the same way preprocess_v1 does
    c = conversation_lib.default_conversation.copy()
    roles = {"human": c.roles[0], "gpt": c.roles[1]}
    src = sources[0]
    if roles[src[0]["from"]] != c.roles[0]:
        src = src[1:]
    c.messages = []
    for j, sentence in enumerate(src):
        role = roles[sentence["from"]]
        c.append_message(role, sentence["value"])
    prompt = c.get_prompt()

    result = preprocess_v1(sources, tokenizer, has_image=has_image)
    labels = result["labels"][0]
    n_valid = (labels != IGNORE_INDEX).sum().item()
    n_total = labels.shape[0]
    return n_valid, n_total, prompt


def trace_masking(sources, has_image=True):
    """Replicate the preprocess_v1 masking loop and print intermediate values."""
    c = conversation_lib.default_conversation.copy()
    roles = {"human": c.roles[0], "gpt": c.roles[1]}
    src = sources[0]
    if roles[src[0]["from"]] != c.roles[0]:
        src = src[1:]
    c.messages = []
    for j, sentence in enumerate(src):
        role = roles[sentence["from"]]
        c.append_message(role, sentence["value"])
    prompt = c.get_prompt()

    sep = c.sep + c.roles[1] + ": "
    if has_image:
        input_ids = tokenizer_image_token(prompt, tokenizer)
    else:
        input_ids = tokenizer(prompt).input_ids
    total_len = sum(1 for t in input_ids if t != tokenizer.pad_token_id)

    print(f"  total_len={total_len} (from tokenizer_image_token)")
    rounds = prompt.split(c.sep2)
    cur_len = 1
    for i, rou in enumerate(rounds):
        if rou == "":
            print(f"  round[{i}]: EMPTY → break")
            break
        parts = rou.split(sep)
        if len(parts) != 2:
            print(f"  round[{i}]: split gives {len(parts)} parts → break")
            break
        parts[0] += sep
        if has_image:
            rl = len(tokenizer_image_token(rou, tokenizer))
            il = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            rl = len(tokenizer(rou).input_ids)
            il = len(tokenizer(parts[0]).input_ids) - 2
        adj = ""
        if i != 0 and not getattr(tokenizer, 'legacy', True) and IS_TOKENIZER_GREATER_THAN_0_14:
            rl -= 1
            il -= 1
            adj = " [adj -1]"
        print(f"  round[{i}]: round_len={rl}{adj}, instr_len={il}, cur_len_before={cur_len}, cur_len_after={cur_len + rl}")
        cur_len += rl
    print(f"  Final cur_len={cur_len}, total_len={total_len}, match={cur_len == total_len}")

# ---------------------------------------------------------------
# Test 1: Synthetic single-turn conversation
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: Synthetic single-turn image conversation")
print("=" * 60)

sources_1turn = [[
    {"from": "human",  "value": "<image>\nWhat do you see in this image?"},
    {"from": "gpt",    "value": "I see a cat sitting on a red chair."},
]]

conv2 = conversation_lib.default_conversation.copy()
conv2.messages = []
conv2.append_message(conv2.roles[0], sources_1turn[0][0]["value"])
conv2.append_message(conv2.roles[1], sources_1turn[0][1]["value"])
prompt = conv2.get_prompt()
print(f"Prompt: {repr(prompt)}")
print()
rounds = prompt.split(conv2.sep2)
for i, r in enumerate(rounds):
    print(f"  round[{i}]: {repr(r[:80])}")
parts = rounds[0].split(sep)
print(f"  Round 0 split by sep={repr(sep)}: len(parts)={len(parts)}")
print()

n_valid, n_total, _ = check_sources(sources_1turn, has_image=True)
print(f"  Non-IGNORE labels: {n_valid} / {n_total}")
if n_valid == 0:
    print("  *** FAILURE: All labels are IGNORE_INDEX ***")
else:
    print("  *** SUCCESS ***")
print()
print("  Round-length trace (single-turn):")
trace_masking(sources_1turn, has_image=True)

# ---------------------------------------------------------------
# Test 2: Synthetic multi-turn conversation
# ---------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: Synthetic multi-turn image conversation (2 Q&A)")
print("=" * 60)

sources_2turn = [[
    {"from": "human",  "value": "<image>\nWhat do you see in this image?"},
    {"from": "gpt",    "value": "I see a cat sitting on a red chair."},
    {"from": "human",  "value": "What color is the chair?"},
    {"from": "gpt",    "value": "The chair is red."},
]]

conv3 = conversation_lib.default_conversation.copy()
conv3.messages = []
for s in sources_2turn[0]:
    role = "USER" if s["from"] == "human" else "ASSISTANT"
    conv3.append_message(role, s["value"])
prompt2 = conv3.get_prompt()
print(f"Prompt: {repr(prompt2[:200])}")
print()
rounds2 = prompt2.split(conv3.sep2)
for i, r in enumerate(rounds2):
    print(f"  round[{i}]: {repr(r[:80])}")
print()

n_valid2, n_total2, _ = check_sources(sources_2turn, has_image=True)
print(f"  Non-IGNORE labels: {n_valid2} / {n_total2}")
if n_valid2 == 0:
    print("  *** FAILURE: All labels are IGNORE_INDEX ***")
else:
    print("  *** SUCCESS ***")
print()
print("  Round-length trace (multi-turn):")
trace_masking(sources_2turn, has_image=True)

# ---------------------------------------------------------------
# Test 3: Real training data (if --data_path provided)
# ---------------------------------------------------------------
if args.data_path is not None:
    import json
    import copy

    print()
    print("=" * 60)
    print(f"TEST 3: Real data from {args.data_path}")
    print("=" * 60)

    with open(args.data_path) as f:
        data = json.load(f)

    all_image_samples = [d for d in data if 'image' in d]

    # Sort by word count (descending) to test the LONGEST samples first
    # (these are processed first by the modality-length sampler in training)
    all_image_samples.sort(
        key=lambda d: sum(len(c['value'].split()) for c in d['conversations']),
        reverse=True
    )
    image_samples = all_image_samples[:args.n_samples]

    print(f"Testing {len(image_samples)} LONGEST image samples (by word count)...")
    print(f"  Longest sample word count: {sum(len(c['value'].split()) for c in image_samples[0]['conversations'])}")
    print(f"  Shortest in this set:      {sum(len(c['value'].split()) for c in image_samples[-1]['conversations'])}")
    print()

    n_fail = 0
    n_success = 0
    for idx, sample in enumerate(image_samples):
        conversations = copy.deepcopy(sample["conversations"])

        # Mimic preprocess_multimodal (mm_use_im_start_end=False)
        DEFAULT_IMAGE_TOKEN = "<image>"
        for sentence in conversations:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN)

        sources = [conversations]
        result = preprocess_v1(sources, tokenizer, has_image=True)
        labels = result["labels"][0]
        input_ids_tok = result["input_ids"][0]
        n_valid = (labels != IGNORE_INDEX).sum().item()
        n_total = labels.shape[0]

        # Simulate the full training pipeline:
        #   1. DataCollator truncates to MODEL_MAX_LENGTH  (BEFORE the model)
        #   2. prepare_inputs_labels_for_multimodal expands image token
        #   3. Model truncates to tokenizer_model_max_length (= MODEL_MAX_LENGTH)
        IMAGE_TOKEN_INDEX = -200
        MODEL_MAX_LENGTH = 2048
        NUM_IMAGE_PATCHES = 576

        # Step 1: data-collator truncation
        input_ids_dc = input_ids_tok[:MODEL_MAX_LENGTH]
        labels_dc = labels[:MODEL_MAX_LENGTH]
        n_total_dc = input_ids_dc.shape[0]

        # Step 2+3: image expansion then model truncation
        img_pos = (input_ids_dc == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if len(img_pos) > 0:
            expanded_len = n_total_dc - 1 + NUM_IMAGE_PATCHES
            labels_expanded = torch.cat([
                labels_dc[:img_pos[0]],
                torch.full((NUM_IMAGE_PATCHES,), IGNORE_INDEX, dtype=labels_dc.dtype),
                labels_dc[img_pos[0]+1:]
            ])
            labels_after_trunc = labels_expanded[:MODEL_MAX_LENGTH]
            n_valid_after_trunc = (labels_after_trunc != IGNORE_INDEX).sum().item()
        else:
            expanded_len = n_total_dc
            n_valid_after_trunc = (labels_dc != IGNORE_INDEX).sum().item()

        if n_valid_after_trunc == 0 and n_valid > 0:
            status = "TRUNC"  # Labels exist but truncated away
            n_fail += 1
        elif n_valid == 0:
            status = "FAIL"
            n_fail += 1
        else:
            status = "OK"
            n_success += 1

        if idx < 5 or n_valid_after_trunc == 0:
            n_turns = len(conversations)
            word_count = sum(len(c['value'].split()) for c in sample['conversations'])
            print(f"  [{status}] sample {idx} (id={sample.get('id','?')}, turns={n_turns}, words={word_count}): "
                  f"preproc={n_valid}/{n_total} tok"
                  f", dc_trunc={n_total_dc}"
                  f", after_expand={n_valid_after_trunc}/{expanded_len}")
            if n_valid_after_trunc == 0 and n_valid > 0:
                img_pos_val = img_pos[0].item() if len(img_pos) > 0 else -1
                first_nonignore = (labels_dc != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                resp_start = first_nonignore[0].item() if len(first_nonignore) > 0 else -1
                print(f"         image_token_pos={img_pos_val}, first_resp_label_in_dc={resp_start}")
                print(f"         first_resp_after_expansion={resp_start + NUM_IMAGE_PATCHES - 1} (limit={MODEL_MAX_LENGTH})")
                print()

    print()
    print(f"Results: {n_success} SUCCESS, {n_fail} FAILURE out of {len(image_samples)} samples")
    if n_fail == 0:
        print("*** All real-data samples PASS — preprocessing is not the issue ***")
    elif n_fail == len(image_samples):
        print("*** ALL real-data samples FAIL — preprocessing is the root cause ***")
    else:
        print(f"*** {n_fail}/{len(image_samples)} samples fail — partial failure ***")

else:
    print()
    print("(Skipping real-data test — pass --data_path to enable)")

print()
print("=" * 60)
print("Also testing text path (has_image=False):")
n_valid_text, n_total_text, _ = check_sources(sources_1turn, has_image=False)
print(f"  Non-IGNORE labels: {n_valid_text} / {n_total_text}")
if n_valid_text == 0:
    print("  *** FAILURE (text path broken) ***")
else:
    print("  *** OK ***")
