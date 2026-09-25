"""Build the Chinese GEC training set: a token-matched random subsample of the MuCGEC training pairs
(Lang-8 + HSK, $TG_DATA/mucgec/MuCGEC_exp_data/train/train.para) in TowerBlocks format.

Pairs are shuffled with seed 42 and taken in order until the total reaches TARGET_TOKENS = 355,441,
the token count of the original Italian GEC file (train+dev, 732 essays) at the time the set was made,
counted with the TowerBase tokenizer in the ChatML format train.py uses. Result: 2,610 pairs.
Unchanged pairs (source == target) are dropped. Output: $TG_DATA/gec_zh/train-gec_zh_mucgec.parquet
"""
import os
import random
import sys

from datasets import Dataset
from transformers import AutoTokenizer

DATA = os.environ.get("TG_DATA", os.path.expanduser("~/data/TowerBlocks"))
MODELS = os.environ.get("TG_MODELS", os.path.expanduser("~/models"))
SRC = os.path.join(DATA, "mucgec/MuCGEC_exp_data/train/train.para")
OUT = os.path.join(DATA, "gec_zh/train-gec_zh_mucgec.parquet")
TARGET_TOKENS = 355_441
SEED = 42

INSTRUCTION = (
    "GEC is a task designed to correct grammatical, orthographic, and other minor "
    "errors in a piece of text, making only the edits necessary to fix mistakes "
    "while preserving the original meaning.\n"
    "Original (Chinese): {orig}\n"
    "Corrected: "
)

if os.path.exists(OUT):
    print(f"{OUT} already built, not overwriting (setup/verify.py checks its contents)")
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "TowerBase-7B"))
tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"], "pad_token": "<pad>"})


def n_tokens(human, gpt):
    text = tok.bos_token + f"<|im_start|>user\n{human}<|im_end|>\n<|im_start|>assistant\n{gpt}<|im_end|>\n"
    return len(tok(text, add_special_tokens=False)["input_ids"])


pairs = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) == 2 and p[0].strip() and p[1].strip() and p[0] != p[1]:
            pairs.append((p[0].strip(), p[1].strip()))
random.Random(SEED).shuffle(pairs)

records, total = [], 0
for src, tgt in pairs:
    human = INSTRUCTION.format(orig=src)
    t = n_tokens(human, tgt)
    if total + t > TARGET_TOKENS:
        break
    total += t
    records.append({
        "conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": tgt}],
        "lang": "zh", "split": "train", "dataset": "mucgec_zh_gec",
        "task": "grammatical_error_correction",
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
Dataset.from_list(records).to_parquet(OUT)
print(f"wrote {len(records)} examples, {total:,} tokens (target {TARGET_TOKENS:,}) -> {OUT}")
