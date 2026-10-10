"""Score a generate.py output file on the Tower paper's GEC sets (eval/data/gec_paper_eval.jsonl) the way
tower-eval scores them (configs/tower_paper/tower_instruct_5_shot.yaml in deep-spin/tower-eval):
  edit rate  TER of the output against the reference, sacrebleu defaults; lower is better. This is the
             paper's main GEC metric (Tables 3 and 24). The unedited source is scored too: a model only
             helps if it beats leaving the text as it is.
  ERRANT     outputs are tokenised with spaCy (<lang>_core_*_sm), aligned to the tokenised source with
             errant_parallel and compared with the gold M2 file by errant_compare (span-based
             correction, F0.5). Official ERRANT only has an English model, so like tower-eval this
             aligns German and Spanish with it too (Table 22).
tower-eval stops generating at the first newline; outputs are cut there before scoring, as in score_tasks.py.
Run with $TG_SCORERS/errant-venv/bin/python (setup/3b_errant.sh).
Writes <run>.gec_paper.scored.jsonl (per sentence, with the TER and ERRANT counts eval/significance.py
resamples), <run>.gec_paper_scores.json (aggregates) and gec_work/<run>/gec_paper.<lang>.m2 (the aligned
outputs ERRANT scored).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

import errant
import sacrebleu
import spacy

from m2_counts import per_sentence_counts

HERE = os.path.dirname(os.path.abspath(__file__))
REF_M2 = {"en": "conll14.en.m2", "de": "fm.de.m2", "es": "cowsl2h.es.m2"}
SPACY = {"en": "en_core_web_sm", "de": "de_core_news_sm", "es": "es_core_news_sm"}  # as tower-eval
BIN = os.path.dirname(sys.executable)  # errant_parallel / errant_compare of this venv
COMPARE_M2 = os.path.join(os.path.dirname(errant.__file__), "commands", "compare_m2.py")
# TowerInstruct-7B in the Tower paper: edit rate (Table 24) and the unedited source ("no edits").
# Table 22 gives ERRANT only as the average over the 3 languages: 0.42 (F0.5 as a fraction).
PAPER_ER = {"en": 13.39, "de": 14.77, "es": 17.23}
PAPER_NO_EDITS = {"en": 13.75, "de": 18.23, "es": 18.00}
PAPER_ERRANT_AVG = 42.0

p = argparse.ArgumentParser()
p.add_argument("--hyp", required=True, help="JSONL from generate.py on data/gec_paper_eval.jsonl")
a = p.parse_args()

rows = [json.loads(l) for l in open(a.hyp, encoding="utf-8")]
print(f"scoring {len(rows)} outputs from {a.hyp}", flush=True)
for r in rows:
    r["multiline"] = "\n" in r["hyp"]
    # "\r" would split the line when ERRANT reads its input files, and break the alignment
    r["hyp"] = r["hyp"].split("\n")[0].replace("\r", " ").strip()
by_lang = defaultdict(list)
for r in rows:
    by_lang[r["lp"]].append(r)
base = a.hyp[:-len(".jsonl")] if a.hyp.endswith(".jsonl") else a.hyp  # results/<run>.gec_paper
run = os.path.basename(base).replace(".gec_paper", "")
work = os.path.join(os.path.dirname(os.path.abspath(a.hyp)), "gec_work", run)
os.makedirs(work, exist_ok=True)

ter = sacrebleu.metrics.TER()


def errant_scores(lang, items):
    """ERRANT P/R/F0.5 (in %) of the outputs, and TP/FP/FN. Also stores each sentence's counts in items."""
    nlp = spacy.load(SPACY[lang], disable=["tagger", "parser", "ner"])
    tok_hyp = [" ".join(t.text for t in doc) for doc in nlp.pipe([r["hyp"] for r in items])]
    hyp_m2 = os.path.join(work, f"gec_paper.{lang}.m2")
    ref_m2 = os.path.join(HERE, "data", "gec_paper_ref", REF_M2[lang])
    with tempfile.TemporaryDirectory() as tmp:
        orig, cor = (os.path.join(tmp, n) for n in ("orig.txt", "cor.txt"))
        with open(orig, "w", encoding="utf-8") as f:
            f.write("\n".join(r["tok_src"] for r in items) + "\n")
        with open(cor, "w", encoding="utf-8") as f:
            f.write("\n".join(tok_hyp) + "\n")
        subprocess.run([os.path.join(BIN, "errant_parallel"), "-orig", orig, "-cor", cor, "-out", hyp_m2],
                       check=True, capture_output=True)
        with open(hyp_m2, encoding="utf-8") as f:
            n_m2 = sum(l.startswith("S ") for l in f)
        assert n_m2 == len(items), f"{lang}: ERRANT aligned {n_m2} sentences, expected {len(items)}"
    out = subprocess.run([os.path.join(BIN, "errant_compare"), "-hyp", hyp_m2, "-ref", ref_m2],
                         check=True, capture_output=True, text=True).stdout
    # the line under "TP FP FN Prec Rec F0.5"
    values = re.findall(r"\d+\.\d+|\d+", out.strip().split("\n")[2])
    tp, fp, fn = (int(v) for v in values[:3])
    counts = per_sentence_counts(COMPARE_M2, hyp_m2, ref_m2)
    assert [sum(c) for c in zip(*counts)] == [tp, fp, fn], f"{lang}: per-sentence ERRANT counts do not add up"
    for r, (s_tp, s_fp, s_fn) in zip(items, counts):
        r["errant_tp"], r["errant_fp"], r["errant_fn"] = s_tp, s_fp, s_fn
    prec, rec, f05 = (100 * float(v) for v in values[3:6])
    return {"TP": tp, "FP": fp, "FN": fn, "P": prec, "R": rec, "F0.5": f05}


scores = {}
for lang in ("en", "de", "es"):
    items = by_lang[lang]
    if not items:
        continue
    refs = [[r["ref"] for r in items]]
    for r, (edits, ref_len) in zip(items, ter._extract_corpus_statistics([r["hyp"] for r in items], refs)):
        r["ter_edits"], r["ter_ref_len"] = edits, ref_len
    scores[lang] = {
        "testset": items[0]["testset"],
        "n": len(items),
        "edit_rate": ter.corpus_score([r["hyp"] for r in items], refs).score,
        "edit_rate_no_edits": ter.corpus_score([r["src"] for r in items], refs).score,
        "errant": errant_scores(lang, items),
        "unchanged_pct": 100 * sum(r["hyp"] == r["src"] for r in items) / len(items),
        "multiline": sum(r["multiline"] for r in items),
    }
    print(f"scored {lang}", flush=True)
langs = list(scores)
scores["average"] = {"edit_rate": sum(scores[l]["edit_rate"] for l in langs) / len(langs),
                     "errant_F0.5": sum(scores[l]["errant"]["F0.5"] for l in langs) / len(langs)}

print("\nGEC on the Tower paper's test sets (5-shot prompts). Paper = TowerInstruct-7B in the Tower paper.")
print("Edit rate = TER against the reference, lower is better; 'no edits' = the uncorrected source.")
print(f"{'lang':5s} {'testset':9s} {'n':>5s} {'edit rate':>9s} {'(paper)':>8s} {'no edits':>9s} {'(paper)':>8s}"
      f" {'ERRANT P':>9s} {'R':>6s} {'F0.5':>6s} {'unch.%':>7s}")
for l in langs:
    s, e = scores[l], scores[l]["errant"]
    print(f"{l:5s} {s['testset']:9s} {s['n']:5d} {s['edit_rate']:9.2f} {PAPER_ER[l]:8.2f} "
          f"{s['edit_rate_no_edits']:9.2f} {PAPER_NO_EDITS[l]:8.2f} {e['P']:9.2f} {e['R']:6.2f} {e['F0.5']:6.2f} "
          f"{s['unchanged_pct']:7.1f}")
avg = scores["average"]
print(f"{'avg':5s} {'':9s} {'':5s} {avg['edit_rate']:9.2f} {sum(PAPER_ER.values()) / 3:8.2f} {'':9s} {'':8s} "
      f"{'':9s} {'':6s} {avg['errant_F0.5']:6.2f}   (paper ERRANT F0.5 average: {PAPER_ERRANT_AVG:.2f})")
print(f"\n{sum(r['multiline'] for r in rows)} outputs had text after a newline (cut off before scoring, as tower-eval does)")

scored_path = base + ".scored.jsonl"
with open(scored_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
scores_path = base + "_scores.json"
with open(scores_path, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2)
print("per-sentence outputs written to", scored_path)
print("aggregate scores written to", scores_path)
