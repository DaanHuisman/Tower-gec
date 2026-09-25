"""Build the GEC evaluation set (no internet needed). Output: eval/data/gec_eval.jsonl with fields
  testset, lp, id, src, refs (list), ref (first reference), prompt
`prompt` is the ChatML-wrapped version of the exact GEC instruction used in training.

  merlin_it_dev  81 Italian essays, MultiGEC-2025 dev (held out of training; test has no public refs)
  mucgec_dev     1137 Chinese sentences, MuCGEC dev, 2-8 references each (gold char-level M2 in
                 $TG_DATA/mucgec/MuCGEC_exp_data/valid/valid.gold.m2.char for ChERRANT)
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TG_DATA", os.path.expanduser("~/data/TowerBlocks"))
OUT = os.path.join(HERE, "data", "gec_eval.jsonl")

GEC = ("GEC is a task designed to correct grammatical, orthographic, and other minor errors in a piece "
       "of text, making only the edits necessary to fix mistakes while preserving the original meaning.\n"
       "Original ({lang}): {src}\nCorrected: ")
CHATML = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def make_prompt(lang, src):
    return CHATML.format(user=GEC.format(lang=lang, src=src))


rows = []

it = pd.read_parquet(f"{DATA}/gec_it/dev-gec_it_merlin.parquet")
for r in it.itertuples():
    rows.append({"testset": "merlin_it_dev", "lp": "it", "id": r.essay_id, "src": r.source,
                 "refs": [r.reference], "ref": r.reference, "prompt": make_prompt("Italian", r.source)})

with open(f"{DATA}/mucgec/eval_repo_data/MuCGEC_dev.txt", encoding="utf-8") as f:
    for line in f:
        idx, src, *refs = line.rstrip("\n").split("\t")
        refs = [src if r == "没有错误" else r for r in refs]  # "no error" -> the source is correct
        rows.append({"testset": "mucgec_dev", "lp": "zh", "id": idx, "src": src,
                     "refs": refs, "ref": refs[0], "prompt": make_prompt("Chinese", src)})

# Guard against train/eval leakage.
train_src = set()
for f in (f"{DATA}/gec_it/train-gec_it_merlin.parquet", f"{DATA}/gec_zh/train-gec_zh_mucgec.parquet"):
    for c in pd.read_parquet(f).conversations:
        train_src.add(c[0]["value"].split("\nOriginal (", 1)[1].split("): ", 1)[1].rsplit("\nCorrected: ", 1)[0])
leaked = [r["id"] for r in rows if r["src"] in train_src]
assert not leaked, f"{len(leaked)} eval sources also appear in training data"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

from collections import Counter
print(f"wrote {len(rows)} examples to {OUT}")
for k, n in sorted(Counter(r["testset"] for r in rows).items()):
    print(f"  {k:14s} {n}")
