"""Step 5: check that this machine has exactly the data, models and tools the reference results used.

    source config.sh && $TG_ENV/bin/python setup/verify.py

Every check prints OK / MISSING / DIFFERENT. Content hashes are over the parsed rows, so they do not
depend on the parquet writer version. The last check runs train.py in dry-run mode and compares the
fingerprint of the 30k TowerBlocks replay subset, which must be identical for everyone.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("TG_DATA", os.path.expanduser("~/data/TowerBlocks"))
MODELS = os.environ.get("TG_MODELS", os.path.expanduser("~/models"))
SCORERS = os.environ.get("TG_SCORERS", os.path.expanduser("~/gec-scorers"))
ENV = os.environ.get("TG_ENV", os.path.expanduser("~/tower-env"))

FILE_MD5 = {  # raw inputs, as downloaded
    "data/train-00000-of-00004.parquet": "0de323b779587a101405375363294aa3",
    "data/train-00001-of-00004.parquet": "f67424615ca52f3460b865a7669dd722",
    "data/train-00002-of-00004.parquet": "31cbf777f3304f7d7c8a9e6e2c1f0602",
    "data/train-00003-of-00004.parquet": "f81e98ce10414927a86afc0ce3e8f668",
    "mucgec/MuCGEC_exp_data/train/train.para": "fdfb9c57e928a4333e2d7fb27a434f9f",
    "mucgec/MuCGEC_exp_data/valid/valid.gold.m2.char": "e1839a6e4f732a9d009622e25285d724",
    "mucgec/eval_repo_data/MuCGEC_dev.txt": "deb53235fa93658b99d44d575df85d75",
    "multigec-2025-participants-1/italian/Merlin/it-merlin-orig-train.md": "c7efe5569ca59a97a3a6fee96162a9b7",
    "multigec-2025-participants-1/italian/Merlin/it-merlin-ref1-train.md": "4b1ef9c5b05b496549bf8f0470626ce3",
    "multigec-2025-participants-1/italian/Merlin/it-merlin-orig-dev.md": "675f74aa28010361855b4586d577fc99",
    "multigec-2025-participants-1/italian/Merlin/it-merlin-ref1-dev.md": "52dde6bde6ffe0a0d119093ef26d0d21",
}
CONTENT_MD5 = {  # built by setup/4_build_data.sh: (rows, hash)
    "gec_it/train-gec_it_merlin.parquet": (651, "4a11064655be5fde43c78345039e6b36"),
    "gec_it/dev-gec_it_merlin.parquet": (81, "23beb417c228aae8f0038d6ec948c66c"),
    "gec_zh/train-gec_zh_mucgec.parquet": (2610, "b2d7f2554a5af4af14a9c93f29e49cde"),
}
EVAL_MD5 = {  # eval/data, built by setup/4_build_data.sh
    "gec_eval.jsonl": ("e16598f3f7d34420e7a55e50bac235b1", True),
    # FLORES/WMT23/TICO come from external mirrors; a mismatch is a warning, not an error.
    "eval_data.jsonl": ("281dabac62b13419256362e7d776055a", False),
    # TowerEval-Data at a pinned revision
    "tasks_eval.jsonl": ("eff2a0c457844f0d31d95e6e76427447", True),
    "gec_paper_eval.jsonl": ("adb0c2152ee5372ed44f3c499bd0b7d8", True),
    "gec_paper_ref/conll14.en.m2": ("3667b3c1e5bc93317ff60733f9a86bb3", True),
    "gec_paper_ref/fm.de.m2": ("a045c3500247b38283d24810b6c90f85", True),
    "gec_paper_ref/cowsl2h.es.m2": ("51ec0f350506cc7680be8ead8b3da09f", True),
}
REPLAY_MD5 = "beea7a1bfd76"

problems = 0


def report(status, what, detail=""):
    global problems
    if status not in ("OK", "WARN"):
        problems += 1
    print(f"  [{status:9s}] {what}" + (f"  ({detail})" if detail else ""))


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_md5(path):
    df = pd.read_parquet(path)
    h = hashlib.md5()
    for r in df[["conversations", "lang", "split", "dataset", "task"]].itertuples(index=False):
        h.update(json.dumps([[dict(t) for t in r.conversations], r.lang, r.split, r.dataset, r.task],
                            ensure_ascii=False).encode())
    return len(df), h.hexdigest()


print(f"Raw data in {DATA}")
for rel, expected in FILE_MD5.items():
    path = os.path.join(DATA, rel)
    if not os.path.exists(path):
        report("MISSING", rel)
    else:
        got = file_md5(path)
        report("OK" if got == expected else "DIFFERENT", rel, "" if got == expected else f"md5 {got}")

print(f"Built GEC training/dev sets in {DATA}")
for rel, (n, expected) in CONTENT_MD5.items():
    path = os.path.join(DATA, rel)
    if not os.path.exists(path):
        report("MISSING", rel, "run setup/4_build_data.sh")
    else:
        got_n, got = content_md5(path)
        report("OK" if got == expected else "DIFFERENT", f"{rel} ({got_n} rows)",
               "" if got == expected else f"content md5 {got}, expected {n} rows / {expected}")

print(f"Evaluation sets in {ROOT}/eval/data")
for name, (expected, strict) in EVAL_MD5.items():
    path = os.path.join(ROOT, "eval", "data", name)
    if not os.path.exists(path):
        report("MISSING", name, "run setup/4_build_data.sh")
    else:
        got = file_md5(path)
        status = "OK" if got == expected else ("DIFFERENT" if strict else "WARN")
        report(status, name, "" if got == expected else f"md5 {got} -- upstream test data changed?")

print("Models and tools")
for rel in ["TowerInstruct-7B/model-00006-of-00006.safetensors", "TowerInstruct-7B/tokenizer.json",
            "TowerBase-7B/tokenizer.json"]:
    report("OK" if os.path.exists(os.path.join(MODELS, rel)) else "MISSING", f"{MODELS}/{rel}")
report("OK" if os.path.exists(os.path.join(SCORERS, "venv", "bin", "errant_parallel")) else "MISSING",
       f"{SCORERS} (GEC scorers)", "" if os.path.exists(SCORERS) else "run setup/3_scorers.sh")
report("OK" if os.path.exists(os.path.join(SCORERS, "errant-venv", "bin", "errant_compare")) else "MISSING",
       f"{SCORERS}/errant-venv (official ERRANT)", "" if os.path.exists(os.path.join(SCORERS, "errant-venv"))
       else "run setup/3b_errant.sh")

print("TowerBlocks replay subset (train.py dry run)")
if os.path.exists(os.path.join(DATA, "data", "train-00003-of-00004.parquet")) and \
        os.path.exists(os.path.join(MODELS, "TowerInstruct-7B", "tokenizer.json")):
    out = subprocess.run(
        [os.path.join(ENV, "bin", "python"), os.path.join(ROOT, "train.py"),
         "--model", os.path.join(MODELS, "TowerInstruct-7B"), "--data", os.path.join(DATA, "data"),
         "--out", os.path.join(ROOT, "logs", "verify_dry_run"), "--replay_samples", "30000",
         "--max_samples", "16", "--dry_run"],
        capture_output=True, text=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    m = re.search(r"md5=(\w+)", out.stdout)
    if not m:
        report("DIFFERENT", "replay subset", "dry run failed:\n" + (out.stdout + out.stderr)[-1500:])
    else:
        report("OK" if m.group(1) == REPLAY_MD5 else "DIFFERENT", f"replay subset md5={m.group(1)}",
               "" if m.group(1) == REPLAY_MD5 else f"expected {REPLAY_MD5}")
else:
    report("MISSING", "replay subset", "needs TowerBlocks and the TowerInstruct tokenizer")

print()
print("All checks passed." if problems == 0 else f"{problems} check(s) failed -- see above.")
sys.exit(1 if problems else 0)
