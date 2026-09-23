"""Build the evaluation set: FLORES-101(-200 subset), WMT23, TICO-19, restricted to
the 10 languages Tower was trained on. Run this on the LOGIN NODE (needs internet).
Output: one JSONL at eval/data/eval_data.jsonl with fields
  testset, lp, src, ref, prompt
`prompt` uses the exact ChatML template Tower was trained/evaluated with.
"""
import json
import os

import pandas as pd
import sacrebleu
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "eval_data.jsonl")

# Tower's 10 supported languages.
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "zh": "Chinese",
              "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "ko": "Korean",
              "it": "Italian", "es": "Spanish"}

PROMPT = ("<|im_start|>user\nTranslate the following {src_name} source text to "
          "{tgt_name}:\n{src_name}: {src}\n{tgt_name}: <|im_end|>\n<|im_start|>assistant\n")


def make_prompt(src_lang, tgt_lang, src_text):
    return PROMPT.format(src_name=LANG_NAMES[src_lang], tgt_name=LANG_NAMES[tgt_lang], src=src_text)


rows = []

# ---- FLORES (devtest, 1012 aligned sentences across all languages) ----
# gsarti/flores_101 mirrors FLORES-101 (subset of the later FLORES-200) and is not
# access-gated, unlike facebook/flores. It covers all 10 Tower languages.
FLORES_CODE = {"en": "eng", "de": "deu", "fr": "fra", "zh": "zho_simpl", "pt": "por",
               "nl": "nld", "ru": "rus", "ko": "kor", "it": "ita", "es": "spa"}
flores_path = hf_hub_download("gsarti/flores_101", "all/devtest/0000.parquet",
                              repo_type="dataset", revision="refs/convert/parquet")
flores = pd.read_parquet(flores_path)
for lang in LANG_NAMES:
    if lang == "en":
        continue
    en_col, xx_col = f"sentence_{FLORES_CODE['en']}", f"sentence_{FLORES_CODE[lang]}"
    for en_text, xx_text in zip(flores[en_col], flores[xx_col]):
        rows.append({"testset": "flores200", "lp": f"en-{lang}", "src": en_text, "ref": xx_text,
                     "prompt": make_prompt("en", lang, en_text)})
        rows.append({"testset": "flores200", "lp": f"{lang}-en", "src": xx_text, "ref": en_text,
                     "prompt": make_prompt(lang, "en", xx_text)})

# ---- WMT23 general MT test set (via sacrebleu), pairs among Tower languages ----
WMT23_PAIRS = ["en-de", "de-en", "en-ru", "ru-en", "en-zh", "zh-en"]
for lp in WMT23_PAIRS:
    src_lang, tgt_lang = lp.split("-")
    src_file = sacrebleu.get_source_file("wmt23", lp)
    ref_files = sacrebleu.get_reference_files("wmt23", lp)
    with open(src_file, encoding="utf-8") as fs, open(ref_files[0], encoding="utf-8") as fr:
        for s, r in zip(fs, fr):
            s, r = s.strip(), r.strip()
            rows.append({"testset": "wmt23", "lp": lp, "src": s, "ref": r,
                         "prompt": make_prompt(src_lang, tgt_lang, s)})

# ---- TICO-19 (humanitarian/COVID domain), English-sourced only ----
TICO_TARGETS = {"fr": "fr", "es": "es-LA", "pt": "pt-BR", "ru": "ru", "zh": "zh"}
for lang, tico_code in TICO_TARGETS.items():
    p = hf_hub_download("gmnlp/tico19", f"test/test.en-{tico_code}.tsv", repo_type="dataset")
    df = pd.read_csv(p, sep="\t")
    for s, r in zip(df["sourceString"], df["targetString"]):
        s, r = str(s).strip(), str(r).strip()
        rows.append({"testset": "tico19", "lp": f"en-{lang}", "src": s, "ref": r,
                     "prompt": make_prompt("en", lang, s)})

with open(OUT, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

from collections import Counter
c = Counter((r["testset"], r["lp"]) for r in rows)
print(f"wrote {len(rows)} examples to {OUT}")
for k in sorted(c):
    print(f"  {k[0]:10s} {k[1]:6s} {c[k]}")
