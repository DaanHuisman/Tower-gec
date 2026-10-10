import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SCORERS = os.environ.get("TG_SCORERS", os.path.expanduser("~/gec-scorers"))
PY = os.path.join(SCORERS, "venv", "bin", "python")
CHERRANT = os.path.join(SCORERS, "MuCGEC", "scorers", "ChERRANT")
MULTIGEC = os.path.join(SCORERS, "multigec-2025", "scripts", "local_eval", "gleu_errant")
DATA = os.environ.get("TG_DATA", os.path.expanduser("~/data/TowerBlocks"))
MUCGEC_GOLD_M2 = f"{DATA}/mucgec/MuCGEC_exp_data/valid/valid.gold.m2.char"
MERLIN_DIR = f"{DATA}/multigec-2025-participants-1/italian/Merlin"

p = argparse.ArgumentParser()
p.add_argument("--hyp", required=True, help="results/<run>.gec.jsonl from generate.py")
a = p.parse_args()

run = os.path.basename(a.hyp).removesuffix(".gec.jsonl").removesuffix(".jsonl")
res_dir = os.path.dirname(os.path.abspath(a.hyp))
work = os.path.join(res_dir, "gec_work", run)
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
rows = [json.loads(l) for l in open(a.hyp, encoding="utf-8")]
env = dict(os.environ, PATH=f"{os.path.dirname(PY)}:{os.environ['PATH']}")


def run_cmd(cmd, cwd):
    out = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return out.stdout


def parse_prf(text):
    """Both ChERRANT and ERRANT print a 'TP FP FN Prec Rec F0.5' header followed by the values."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.split()[:3] == ["TP", "FP", "FN"]:
            tp, fp, fn, prec, rec, f = lines[i + 1].split()
            return {"TP": int(tp), "FP": int(fp), "FN": int(fn),
                    "P": 100 * float(prec), "R": 100 * float(rec), "F0.5": 100 * float(f)}
    sys.exit("could not find P/R/F in scorer output:\n" + text[-2000:])


def clean(text):
    return " ".join(text.split())  # no tabs/newlines inside a line-based file


scores = {}

# ---- Chinese: ChERRANT (char level) ----
zh = sorted((r for r in rows if r["testset"] == "mucgec_dev"), key=lambda r: int(r["id"]))
if zh:
    gold_src = [l[2:].replace(" ", "").strip() for l in open(MUCGEC_GOLD_M2, encoding="utf-8") if l.startswith("S ")]
    assert [r["src"].replace(" ", "") for r in zh] == gold_src, "hypotheses not aligned with gold M2"
    para = os.path.join(work, "mucgec_dev.para")
    with open(para, "w", encoding="utf-8") as f:
        for r in zh:
            hyp = clean(r["hyp"]) or r["src"]  # ChERRANT cannot align an empty hypothesis
            f.write(f"{r['id']}\t{r['src']}\t{hyp}\n")
    hyp_m2 = os.path.join(work, "mucgec_dev.hyp.m2.char")
    run_cmd([PY, "parallel_to_m2.py", "-f", para, "-o", hyp_m2, "-g", "char", "-s", "all"], CHERRANT)
    out = run_cmd([PY, "compare_m2_for_evaluation.py", "-hyp", hyp_m2, "-ref", MUCGEC_GOLD_M2], CHERRANT)
    scores["mucgec_dev"] = {"metric": "ChERRANT char", "n": len(zh), **parse_prf(out)}

# ---- Italian: official MultiGEC-2025 GLEU + ERRANT ----
it = [r for r in rows if r["testset"] == "merlin_it_dev"]
if it:
    sys.path.insert(0, MULTIGEC)
    from utils_transform_markdown_to_one_essay_per_line import md_to_dict

    ref, res, out_dir = (os.path.join(work, d) for d in ("ref", "res", "out"))
    for d in (ref, res, out_dir):
        os.makedirs(d)
    shutil.copy(f"{MERLIN_DIR}/it-merlin-orig-dev.md", ref)
    # The official script expects ref/*.tmp (one essay per line) and the reference M2 to exist already;
    # build them the same way it builds hypothesis files (syntok pretokenisation, newlines -> tabs, orig order).
    orig = md_to_dict(open(f"{MERLIN_DIR}/it-merlin-orig-dev.md", encoding="utf-8").read())
    ref1 = md_to_dict(open(f"{MERLIN_DIR}/it-merlin-ref1-dev.md", encoding="utf-8").read())
    for name, d in (("orig", orig), ("ref1", ref1)):
        with open(f"{ref}/it-merlin-{name}-dev.tmp", "w", encoding="utf-8") as f:
            for eid in orig:
                f.write(d[eid].replace("\n", "\t") + "\n")
    run_cmd(["errant_parallel", "-orig", f"{ref}/it-merlin-orig-dev.tmp", "-cor", f"{ref}/it-merlin-ref1-dev.tmp",
             "-out", f"{ref}/it-merlin-ref1-dev.m2", "-lang", "it"], MULTIGEC)
    by_id = {r["id"]: r for r in it}
    assert set(by_id) == set(orig), "hypothesis essay ids differ from the dev set"
    with open(f"{res}/it-merlin-hypo1-dev.md", "w", encoding="utf-8") as f:
        for eid in orig:
            f.write(f"### essay_id = {eid}\n{by_id[eid]['hyp'].strip() or '!'}\n\n")
    run_cmd([PY, "gleu_errant_evaluation.py", work, out_dir, "dev"], MULTIGEC)
    txt = open(f"{out_dir}/scores.txt", encoding="utf-8").read()
    gleu = re.search(r"GLEU = (\S+)", txt).group(1)
    prf = {k: float(re.search(rf"{k} = (\S+)", txt).group(1)) * 100 for k in ("precision", "recall", "f0.5")}
    scores["merlin_it_dev"] = {"metric": "MultiGEC-2025 GLEU + ERRANT", "n": len(it), "GLEU": float(gleu),  # gleu already reports 0-100
                               "P": prf["precision"], "R": prf["recall"], "F0.5": prf["f0.5"]}

# ---- report ----
for ts, s in scores.items():
    n_unchanged = sum(r["hyp"].strip() == r["src"].strip() for r in rows if r["testset"] == ts)
    s["unchanged_pct"] = 100 * n_unchanged / s["n"]
print(f"\nGEC scores for {run}")
print(f"{'testset':14s} {'n':>5s} {'GLEU':>6s} {'P':>6s} {'R':>6s} {'F0.5':>6s} {'unchanged%':>10s}")
for ts, s in scores.items():
    gleu = f"{s['GLEU']:6.2f}" if "GLEU" in s else "     -"
    print(f"{ts:14s} {s['n']:5d} {gleu} {s['P']:6.2f} {s['R']:6.2f} {s['F0.5']:6.2f} {s['unchanged_pct']:10.1f}")
out_path = os.path.join(res_dir, f"{run}.gec_scores.json")
json.dump(scores, open(out_path, "w"), indent=2)
print("\nwritten to", out_path)
