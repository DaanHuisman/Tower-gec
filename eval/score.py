"""Score a generate.py output file with COMET-22 (Unbabel/wmt22-comet-da) and chrF,
aggregate by (testset, direction) and (testset, language pair), and print alongside
the TowerInstruct-7B numbers reported in Table 1 of the Tower paper (arXiv:2402.17733)
for a direct sanity check.
"""
import argparse
import json
from collections import defaultdict

import sacrebleu
from comet import download_model, load_from_checkpoint

p = argparse.ArgumentParser()
p.add_argument("--hyp", required=True, help="JSONL from generate.py")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--gpus", type=int, default=1)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.hyp, encoding="utf-8")]
print(f"scoring {len(rows)} segments from {a.hyp}", flush=True)

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
comet_in = [{"src": r["src"], "mt": r["hyp"], "ref": r["ref"]} for r in rows]
comet_out = model.predict(comet_in, batch_size=a.batch_size, gpus=a.gpus)
for r, s in zip(rows, comet_out["scores"]):
    r["comet22"] = s * 100

# ---- aggregate ----
def direction(lp):
    src, tgt = lp.split("-")
    return "en->xx" if src == "en" else "xx->en"

by_lp = defaultdict(list)
by_dir = defaultdict(list)
for r in rows:
    key_lp = (r["testset"], r["lp"])
    key_dir = (r["testset"], direction(r["lp"]))
    by_lp[key_lp].append(r)
    by_dir[key_dir].append(r)

def avg_comet(items):
    return sum(x["comet22"] for x in items) / len(items)

def chrf(items):
    return sacrebleu.corpus_chrf([x["hyp"] for x in items], [[x["ref"] for x in items]]).score

print("\nPer language pair:")
print(f"{'testset':10s} {'lp':7s} {'n':>6s} {'COMET-22':>9s} {'chrF':>7s}")
for k in sorted(by_lp):
    items = by_lp[k]
    print(f"{k[0]:10s} {k[1]:7s} {len(items):6d} {avg_comet(items):9.2f} {chrf(items):7.2f}")

# TowerInstruct-7B numbers from Table 1 of the paper, for reference.
PAPER_T7B = {
    ("flores200", "en->xx"): 88.51, ("flores200", "xx->en"): 88.27,
    ("wmt23", "en->xx"): 84.28, ("wmt23", "xx->en"): 82.77,
    ("tico19", "en->xx"): 87.01,
}

print("\nAggregated by direction (COMET-22), vs. paper's TowerInstruct-7B (Table 1):")
print(f"{'testset':10s} {'dir':8s} {'n':>6s} {'ours':>8s} {'paper':>8s} {'delta':>7s}")
for k in sorted(by_dir):
    items = by_dir[k]
    ours = avg_comet(items)
    paper = PAPER_T7B.get(k)
    paper_s = f"{paper:8.2f}" if paper is not None else "     n/a"
    delta_s = f"{ours - paper:+7.2f}" if paper is not None else "    n/a"
    print(f"{k[0]:10s} {k[1]:8s} {len(items):6d} {ours:8.2f} {paper_s} {delta_s}")

out_path = a.hyp.replace(".jsonl", ".scored.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\nper-segment scores written to", out_path)
