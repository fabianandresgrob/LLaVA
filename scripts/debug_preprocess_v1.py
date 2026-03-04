#!/usr/bin/env python3
"""
Standalone diagnostic for the 'tokenization mismatch: 1 vs. N' bug.

Run on HMGU (no GPU needed):
    cd /ictstr01/home/eml/fabian.grob/projects/LLaVA
    conda activate llava
    python scripts/debug_preprocess_v1.py

Checks whether preprocess_v1 correctly masks labels for image-containing samples.
"""

import sys, os

# Make sure llava is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from llava import conversation as conversation_lib
from llava.train.train import preprocess_v1, IS_TOKENIZER_GREATER_THAN_0_14
from llava.mm_utils import tokenizer_image_token
import tokenizers

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

# Build a sample conversation (mimics preprocess_multimodal output)
sources = [[
    {"from": "human",  "value": "<image>\nWhat do you see in this image?"},
    {"from": "gpt",    "value": "I see a cat sitting on a red chair."},
]]

# Show the conversation prompt
conv2 = conversation_lib.default_conversation.copy()
conv2.messages = []
conv2.append_message(conv2.roles[0], sources[0][0]["value"])
conv2.append_message(conv2.roles[1], sources[0][1]["value"])
prompt = conv2.get_prompt()
print("Conversation prompt:")
print(f"  {repr(prompt)}")
print()
print("  Split by sep2:")
rounds = prompt.split(conv2.sep2)
for i, r in enumerate(rounds):
    print(f"  round[{i}]: {repr(r[:80])}")
print()
print(f"  Round 0 split by sep={repr(sep)}:")
parts = rounds[0].split(sep)
print(f"  len(parts)={len(parts)}")
for i, p in enumerate(parts):
    print(f"  parts[{i}]: {repr(p[:80])}")
print()

# Run preprocess_v1 and check labels
print("Running preprocess_v1 with has_image=True ...")
result = preprocess_v1(sources, tokenizer, has_image=True)
input_ids = result["input_ids"][0]
labels = result["labels"][0]
from llava.constants import IGNORE_INDEX
n_valid = (labels != IGNORE_INDEX).sum().item()
n_total = labels.shape[0]
print(f"  Sequence length:    {n_total}")
print(f"  Non-IGNORE labels:  {n_valid} / {n_total}")
if n_valid == 0:
    print("  *** FAILURE: All labels are IGNORE_INDEX — loss will be 0! ***")
else:
    print("  *** SUCCESS: Labels correctly masked ***")

# Also test without image (should always work)
print()
print("Running preprocess_v1 with has_image=False ...")
result_text = preprocess_v1(sources, tokenizer, has_image=False)
labels_text = result_text["labels"][0]
n_valid_text = (labels_text != IGNORE_INDEX).sum().item()
print(f"  Non-IGNORE labels: {n_valid_text} / {labels_text.shape[0]}")
if n_valid_text == 0:
    print("  *** FAILURE (text path also broken) ***")
else:
    print("  *** OK (text path works as expected) ***")
