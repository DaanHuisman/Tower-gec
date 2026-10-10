"""Paired bootstrap significance tests for the comparisons in our results:
  control - official   the effect of the extra training on its own
  it - control         the effect of the Italian GEC data
  zh - control         the effect of the Chinese GEC data
  it - official, zh - official

Each test set is resampled with replacement 1,000 times. The resampling is paired (both systems are scored
on the same resampled items), and every metric is recomputed from the resampled per-item statistics, so
corpus-level metrics (F0.5, GLEU, F1, TER) are bootstrapped as corpus-level metrics, not as averages of
sentence scores. Averages over languages or language pairs resample each one separately. Reported: the
difference B - A, its 95% percentile interval, and a two-sided p-value (twice the share of resampled
differences on the other side of zero). The Tower paper also bootstraps its corpus-level metrics (Koehn 2004).

    $TG_SCORERS/venv/bin/python eval/significance.py

Runs in the scorer venv (setup/3_scorers.sh), which has the GLEU package used for the Italian scores; the
ERRANT / ChERRANT comparison code is loaded from the scorer checkouts. Reads the per-item outputs in
results/ (<run>.*.jsonl and gec_work/<run>/), first checks that they reproduce the reported scores, and
writes results/SIGNIFICANCE.md and results/significance.json.
"""
import json
import os
import zlib
from collections import defaultdict

import numpy as np
from gleu.aggreg import make_n_accum
from gleu.count import set_tokenization

from m2_counts import per_sentence_counts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SCORERS = os.environ.get("TG_SCORERS", os.path.expanduser("~/gec-scorers"))
DATA = os.environ.get("TG_DATA", os.path.expanduser("~/data/TowerBlocks"))
ERRANT_COMPARE = f"{SCORERS}/multigec-2025/scripts/local_eval/gleu_errant/errant/errant/commands/compare_m2.py"
CHERRANT_COMPARE = f"{SCORERS}/MuCGEC/scorers/ChERRANT/compare_m2_for_evaluation.py"
MUCGEC_GOLD_M2 = f"{DATA}/mucgec/MuCGEC_exp_data/valid/valid.gold.m2.char"

RUNS = {"official": "towerinstruct_7b_official", "control": "gec_control", "it": "gec_it", "zh": "gec_zh"}
PAIRS = [("official", "control"), ("control", "it"), ("control", "zh"), ("official", "it"), ("official", "zh")]
N_BOOT, SEED, ALPHA = 1000, 12345, 0.05


# ---- metrics: computed from summed per-item statistics, on the last axis ----
def mean(s):  # stats: [score, 1]
    return s[..., 0] / s[..., 1]


def errant_prf(s):  # stats: [tp, fp, fn]; ERRANT's computeFScore conventions
    tp, fp, fn = s[..., 0], s[..., 1], s[..., 2]
    p = np.where(fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    r = np.where(fn > 0, tp / np.maximum(tp + fn, 1), 1.0)
    f = np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    return 100 * p, 100 * r, 100 * f


def precision(s):
    return errant_prf(s)[0]


def recall(s):
    return errant_prf(s)[1]


def f05(s):
    return errant_prf(s)[2]


def ner_f1(s):  # stats: [correct, gold, predicted]; as score_tasks.py
    c, g, p = s[..., 0], s[..., 1], s[..., 2]
    prec = np.where(p > 0, c / np.maximum(p, 1), 0.0)
    rec = np.where(g > 0, c / np.maximum(g, 1), 0.0)
    return 100 * np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)


def ter(s):  # stats: [edits, reference length]; as sacrebleu's TER
    edits, ref_len = s[..., 0], s[..., 1]
    return 100 * np.where(ref_len > 0, edits / np.maximum(ref_len, 1), (edits > 0) * 1.0)


def gleu(s):  # stats: [match, penalty, denominator] for n = 1..4, then [ref length, hyp length]; as gleu 1.1.0
    logps = []
    for n in range(4):
        match, penal, denom = s[..., 3 * n], s[..., 3 * n + 1], s[..., 3 * n + 2]
        p = np.where(denom > 0, (match - np.minimum(match, penal)) / np.maximum(denom, 1), 1.0)
        with np.errstate(divide="ignore"):
            logps.append(np.log(p))
    rlen, hlen = s[..., 12], s[..., 13]
    with np.errstate(divide="ignore", invalid="ignore"):
        logbp = np.where(hlen == 0, np.where(rlen == 0, 0.0, -np.inf),
                         np.where(rlen < hlen, 0.0, 1.0 - rlen / np.maximum(hlen, 1)))
    return 100 * np.exp(logbp + np.mean(logps, axis=0))


# ---- per-item statistics for each run ----
def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def by_key(rows, key, stats):
    """{group: (n, k) array} of stats(row) for the rows of each group."""
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(stats(r))
    return {k: np.array(v, dtype=float) for k, v in sorted(out.items())}


def italian_gec(run):
    w = os.path.join(RESULTS, "gec_work", run)
    counts = per_sentence_counts(ERRANT_COMPARE, f"{w}/res/it-merlin-hypo1-dev.m2", f"{w}/ref/it-merlin-ref1-dev.m2")
    # GLEU as the MultiGEC script runs it: gleu -s orig -r ref1 -o hyp -n 4 -t word (one reference)
    set_tokenization("word")
    src, ref, hyp = ([l.rstrip("\n") for l in open(f, encoding="utf-8")] for f in
                     (f"{w}/ref/it-merlin-orig-dev.tmp", f"{w}/ref/it-merlin-ref1-dev.tmp", f"{w}/res/it-merlin-hypo1-dev.tmp"))
    gleu_stats = []
    for s, r, h in zip(src, ref, hyp):
        accums = [a.rectify() for a in make_n_accum(4, s, r, h)]
        gleu_stats.append([v for a in accums for v in (a.match, a.penal, a.denom)] + [len(r.split()), len(h.split())])
    return np.array(counts, dtype=float), np.array(gleu_stats, dtype=float)


def chinese_gec(run):
    hyp_m2 = os.path.join(RESULTS, "gec_work", run, "mucgec_dev.hyp.m2.char")
    return np.array(per_sentence_counts(CHERRANT_COMPARE, hyp_m2, MUCGEC_GOLD_M2), dtype=float)


def direction(lp):
    return "en→xx" if lp.startswith("en-") else "xx→en"


# ---- tests ----
class Test:
    """One metric on one test set: strata is a list of {run: (n, k) stats}; with several strata the
    metric is averaged over them (each stratum resampled on its own)."""

    def __init__(self, section, name, metric, strata, reported=None, lower_is_better=False):
        self.section, self.name, self.metric, self.strata = section, name, metric, strata
        self.reported, self.lower_is_better = reported or {}, lower_is_better

    def value(self, run):
        return float(np.mean([self.metric(st[run].sum(axis=0)) for st in self.strata]))

    def bootstrap(self, a, b):
        rng = np.random.default_rng([SEED, zlib.crc32(f"{self.section}|{self.name}|{a}|{b}".encode())])
        deltas = np.zeros(N_BOOT)
        for st in self.strata:
            xa, xb = st[a], st[b]
            assert xa.shape == xb.shape
            for start in range(0, N_BOOT, 100):  # chunks keep memory small for the large MT sets
                idx = rng.integers(0, len(xa), size=(min(100, N_BOOT - start), len(xa)))
                deltas[start:start + len(idx)] += self.metric(xb[idx].sum(axis=1)) - self.metric(xa[idx].sum(axis=1))
        deltas /= len(self.strata)
        observed = self.value(b) - self.value(a)
        low, high = np.percentile(deltas, [2.5, 97.5])
        p = min(1.0, 2 * min((deltas <= 0).sum() + 1, (deltas >= 0).sum() + 1) / (N_BOOT + 1))
        return {"delta": observed, "ci": [float(low), float(high)], "p": p}


tests = []
sections = []


def add(section, *new_tests):
    if section not in sections:
        sections.append(section)
    tests.extend(new_tests)


def gec_reported(run, testset, key):
    path = os.path.join(RESULTS, f"{run}.gec_scores.json")
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "reference", f"{run}.gec_scores.json")
    return json.load(open(path))[testset][key]


print("loading per-item outputs ...", flush=True)
# Italian and Chinese GEC (our test sets)
it_counts, it_gleu, zh_counts = {}, {}, {}
for short, run in RUNS.items():
    it_counts[short], it_gleu[short] = italian_gec(run)
    zh_counts[short] = chinese_gec(run)
S = "Italian GEC (MultiGEC-2025 dev, 81 essays)"
add(S, Test(S, "GLEU", gleu, [it_gleu], {r: gec_reported(RUNS[r], "merlin_it_dev", "GLEU") for r in RUNS}),
    *[Test(S, name, fn, [it_counts], {r: gec_reported(RUNS[r], "merlin_it_dev", key) for r in RUNS})
      for name, key, fn in (("F0.5", "F0.5", f05), ("Precision", "P", precision), ("Recall", "R", recall))])
S = "Chinese GEC (MuCGEC dev, 1,137 sentences)"
add(S, *[Test(S, name, fn, [zh_counts], {r: gec_reported(RUNS[r], "mucgec_dev", key) for r in RUNS})
         for name, key, fn in (("F0.5", "F0.5", f05), ("Precision", "P", precision), ("Recall", "R", recall))])

# Translation: COMET-22 per segment
mt = {short: jsonl(os.path.join(RESULTS, f"{run}.scored.jsonl")) for short, run in RUNS.items()}
mt_dir = {r: by_key(mt[r], lambda x: (x["testset"], direction(x["lp"])), lambda x: [x["comet22"], 1]) for r in RUNS}
mt_lp = {r: by_key(mt[r], lambda x: (x["testset"], x["lp"]), lambda x: [x["comet22"], 1]) for r in RUNS}
S = "Translation, COMET-22"
add(S, *[Test(S, f"{ts} {d}", mean, [{r: mt_dir[r][(ts, d)] for r in RUNS}])
         for ts, d in mt_dir["official"]])
S = "Translation, COMET-22 per language pair"
add(S, *[Test(S, f"{ts} {lp}", mean, [{r: mt_lp[r][(ts, lp)] for r in RUNS}]) for ts, lp in mt_lp["official"]])

# APE and NER
tasks = {short: jsonl(os.path.join(RESULTS, f"{run}.tasks.scored.jsonl")) for short, run in RUNS.items()}
tasks_scores = {short: json.load(open(os.path.join(RESULTS, f"{run}.tasks_scores.json"))) for short, run in RUNS.items()}
ape = {r: by_key([x for x in tasks[r] if x["task"] == "ape"], lambda x: x["lp"], lambda x: [x["comet22"], 1]) for r in RUNS}
ner = {r: by_key([x for x in tasks[r] if x["task"] == "ner"], lambda x: x["lp"],
                 lambda x: [x["ner_correct"], x["ner_gold"], x["ner_pred"]]) for r in RUNS}
S = "APE (WMT23, post-editing NLLB-3.3B), COMET-22"
add(S, Test(S, "average of 6 pairs", mean, [{r: ape[r][lp] for r in RUNS} for lp in ape["official"]],
            {r: tasks_scores[r]["ape"]["average"]["post-edit"]["COMET-22"] for r in RUNS}),
    *[Test(S, lp, mean, [{r: ape[r][lp] for r in RUNS}],
           {r: tasks_scores[r]["ape"][lp]["post-edit"]["COMET-22"] for r in RUNS}) for lp in ape["official"]])
S = "NER (MultiCoNER II), F1"
add(S, Test(S, "average of 7 languages", ner_f1, [{r: ner[r][l] for r in RUNS} for l in ner["official"]],
            {r: tasks_scores[r]["ner"]["average"]["F1"] for r in RUNS}),
    *[Test(S, l, ner_f1, [{r: ner[r][l] for r in RUNS}], {r: tasks_scores[r]["ner"][l]["F1"] for r in RUNS})
      for l in ner["official"]])

# The Tower paper's GEC sets: needs the per-sentence statistics score_gec_paper.py writes
paper_files = {short: os.path.join(RESULTS, f"{run}.gec_paper.scored.jsonl") for short, run in RUNS.items()}
skipped = []
if all(os.path.exists(f) for f in paper_files.values()):
    paper = {r: jsonl(f) for r, f in paper_files.items()}
    paper_scores = {short: json.load(open(os.path.join(RESULTS, f"{run}.gec_paper_scores.json")))
                    for short, run in RUNS.items()}
    if all("ter_edits" in rows[0] for rows in paper.values()):
        er = {r: by_key(paper[r], lambda x: x["lp"], lambda x: [x["ter_edits"], x["ter_ref_len"]]) for r in RUNS}
        pe = {r: by_key(paper[r], lambda x: x["lp"], lambda x: [x["errant_tp"], x["errant_fp"], x["errant_fn"]])
              for r in RUNS}
        langs = [l for l in ("en", "de", "es") if l in er["official"]]
        S = "Tower paper's GEC sets (5-shot), edit rate (lower is better)"
        add(S, Test(S, "average of 3 languages", ter, [{r: er[r][l] for r in RUNS} for l in langs],
                    {r: paper_scores[r]["average"]["edit_rate"] for r in RUNS}, lower_is_better=True),
            *[Test(S, l, ter, [{r: er[r][l] for r in RUNS}], {r: paper_scores[r][l]["edit_rate"] for r in RUNS},
                   lower_is_better=True) for l in langs])
        S = "Tower paper's GEC sets (5-shot), ERRANT F0.5"
        add(S, Test(S, "average of 3 languages", f05, [{r: pe[r][l] for r in RUNS} for l in langs],
                    {r: paper_scores[r]["average"]["errant_F0.5"] for r in RUNS}),
            *[Test(S, l, f05, [{r: pe[r][l] for r in RUNS}], {r: paper_scores[r][l]["errant"]["F0.5"] for r in RUNS})
              for l in langs])
    else:
        skipped.append("the Tower paper's GEC sets (rescore them with the current eval/score_gec_paper.py)")
else:
    skipped.append("the Tower paper's GEC sets (no results/<run>.gec_paper.scored.jsonl yet: ./submit.sh gec_paper)")

# ---- check that the per-item statistics reproduce the reported scores ----
mt_reported = {}
for short, run in RUNS.items():
    path = os.path.join(RESULTS, f"{run}.summary.txt")
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "reference", f"{run}.summary.txt")
    for line in open(path):
        parts = line.split()
        if len(parts) == 6 and parts[1] in ("en->xx", "xx->en"):  # aggregated rows: testset dir n ours paper delta
            mt_reported[(short, f"{parts[0]} {parts[1].replace('->', '→')}")] = float(parts[3])
        elif len(parts) == 5 and "-" in parts[1] and parts[0] in ("flores200", "wmt23", "tico19"):
            mt_reported[(short, f"{parts[0]} {parts[1]}")] = float(parts[3])
for t in tests:
    if t.section.startswith("Translation"):
        t.reported = {r: mt_reported[(r, t.name)] for r in RUNS}
mismatches = [(t.section, t.name, r, t.value(r), t.reported[r]) for t in tests for r in RUNS
              if abs(t.value(r) - t.reported[r]) > 0.011]
if mismatches:
    for m in mismatches:
        print("MISMATCH", m)
    raise SystemExit("per-item statistics do not reproduce the reported scores -- not testing")
print(f"per-item statistics reproduce all {len(tests) * len(RUNS)} reported scores", flush=True)

# ---- bootstrap ----
results = []
for i, t in enumerate(tests):
    row = {"section": t.section, "test": t.name, "lower_is_better": t.lower_is_better,
           "values": {r: t.value(r) for r in RUNS}, "comparisons": {}}
    for a, b in PAIRS:
        row["comparisons"][f"{b} - {a}"] = t.bootstrap(a, b)
    results.append(row)
    print(f"  [{i + 1}/{len(tests)}] {t.section}: {t.name}", flush=True)


# ---- report ----
def cell(c):
    floor = 2 / (N_BOOT + 1)  # no resample on the other side of zero: the smallest p-value N_BOOT can show
    p = f"p≤{floor:.3f}".replace("0.", ".") if c["p"] <= floor else f"p={c['p']:.3f}".replace("0.", ".")
    text = f"{c['delta']:+.2f} ({p})"
    return f"**{text}**" if c["p"] < ALPHA else text


pair_names = [f"{b} - {a}" for a, b in PAIRS]
md = ["# Significance tests", "",
      f"Paired bootstrap, {N_BOOT} resamples per test (seed {SEED}), produced by `eval/significance.py`. Each cell "
      "is the difference in score (second model minus first) with its two-sided p-value. **Bold** = significant at "
      f"p < {ALPHA}: the 95% interval of the difference excludes zero. With {N_BOOT} resamples the smallest p-value "
      f"that can be shown is {2 / (N_BOOT + 1):.3f} (written ≤). Intervals are in `significance.json`.", "",
      "- **control - official**: the effect of the extra training on its own",
      "- **it - control**, **zh - control**: the effect of adding Italian or Chinese GEC data",
      "- **it - official**, **zh - official**: both together", "",
      f"There are {len(results) * len(PAIRS)} tests here, so at p < {ALPHA} a few would come out significant by "
      "chance alone. Treat isolated p-values just under 0.05 with care, especially among the per-pair rows.", "",
      "The bootstrap only covers the choice of test sentences. It does not cover the run-to-run variation of "
      "training itself (one run per condition; two identical training runs differed by 0.40 Italian F0.5, see "
      "RESULTS.txt). A small difference can be significant here and still be training noise, so read "
      "significant differences much smaller than that, such as most translation and APE ones, with care.", ""]
if skipped:
    md += ["Not included yet: " + "; ".join(skipped) + ".", ""]
for section in sections:
    rows = [r for r in results if r["section"] == section]
    note = " For edit rate, a negative difference is an improvement." if rows[0]["lower_is_better"] else ""
    md += [f"## {section}", ""]
    if note:
        md += [note.strip(), ""]
    md += ["| | " + " | ".join(RUNS) + " | " + " | ".join(pair_names) + " |",
           "|---" * (1 + len(RUNS) + len(PAIRS)) + "|"]
    for r in rows:
        md.append(f"| {r['test']} | " + " | ".join(f"{r['values'][x]:.2f}" for x in RUNS) + " | "
                  + " | ".join(cell(r["comparisons"][n]) for n in pair_names) + " |")
    md.append("")

with open(os.path.join(RESULTS, "SIGNIFICANCE.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
with open(os.path.join(RESULTS, "significance.json"), "w", encoding="utf-8") as f:
    json.dump({"n_bootstrap": N_BOOT, "seed": SEED, "alpha": ALPHA, "tests": results}, f, indent=1)
print("\n".join(md))
print("written to results/SIGNIFICANCE.md and results/significance.json")
