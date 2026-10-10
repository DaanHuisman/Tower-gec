import argparse
import json
import re
from collections import defaultdict

import sacrebleu
from comet import download_model, load_from_checkpoint

NER_TYPES = {"Person", "Location", "Group", "Product", "CreativeWorks", "Medical"}
TUPLE = re.compile(r'\("(.*?)", "(.*?)"\)')  # tower-eval's parser for [("token", "tag"), ...]

p = argparse.ArgumentParser()
p.add_argument("--hyp", required=True, help="JSONL from generate.py on data/tasks_eval.jsonl")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--gpus", type=int, default=1)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.hyp, encoding="utf-8")]
print(f"scoring {len(rows)} outputs from {a.hyp}", flush=True)
for r in rows:
    r["multiline"] = "\n" in r["hyp"]
    r["hyp"] = r["hyp"].split("\n")[0].strip()
ape = [r for r in rows if r["task"] == "ape"]
ner = [r for r in rows if r["task"] == "ner"]
scores = {"ape": {}, "ner": {}}


def mean(xs):
    return sum(xs) / len(xs)


def group(items):
    by_lp = defaultdict(list)
    for r in items:
        by_lp[r["lp"]].append(r)
    return dict(sorted(by_lp.items()))


# ---- APE ----
if ape:
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    comet_in = [{"src": r["src"], "mt": r[field], "ref": r["ref"]} for field in ("hyp", "mt") for r in ape]
    comet_out = model.predict(comet_in, batch_size=a.batch_size, gpus=a.gpus)["scores"]
    for r, s_hyp, s_mt in zip(ape, comet_out[:len(ape)], comet_out[len(ape):]):
        r["comet22"], r["comet22_nllb"] = s_hyp * 100, s_mt * 100

    ter = sacrebleu.metrics.TER(asian_support=True)
    ter_zh = sacrebleu.metrics.TER(asian_support=True, normalized=True)

    def ape_metrics(items, field, comet_key):
        hyps, refs = [x[field] for x in items], [[x["ref"] for x in items]]
        ter_metric = ter_zh if items[0]["lp"].endswith("-zh") else ter
        return {"COMET-22": mean([x[comet_key] for x in items]),
                "chrF": sacrebleu.corpus_chrf(hyps, refs).score,
                "TER": ter_metric.corpus_score(hyps, refs).score}

    for lp, items in group(ape).items():
        scores["ape"][lp] = {
            "n": len(items),
            "post-edit": ape_metrics(items, "hyp", "comet22"),
            "nllb": ape_metrics(items, "mt", "comet22_nllb"),
            "unchanged_pct": 100 * mean([x["hyp"] == x["mt"] for x in items]),
            "multiline": sum(x["multiline"] for x in items),
        }


# ---- NER ----
def tags(answer):
    # tags outside the 6 types become O, as in tower-eval's filter_tags
    return [t if t[2:] in NER_TYPES else "O" for _, t in TUPLE.findall(answer)]


def chunk_end(prev, tag):  # conlleval's is_chunk_end
    if prev == "O":
        return False
    if tag == "O" or prev[2:] != tag[2:]:
        return True
    return tag[0] in "BS" or prev[0] in "ES"


def chunk_start(prev, tag):  # conlleval's is_chunk_start
    if tag == "O":
        return False
    if prev == "O" or prev[2:] != tag[2:]:
        return True
    return tag[0] in "BS" or prev[0] in "ES"


def chunks(seq):
    """Entities as (start, end, type), with conlleval's rules for where a chunk starts and ends."""
    out, start, prev = set(), None, "O"
    for i, tag in enumerate(seq + ["O"]):
        if start is not None and chunk_end(prev, tag):
            out.add((start, i - 1, prev[2:]))
            start = None
        if chunk_start(prev, tag):
            start = i
        prev = tag
    return out


for lp, items in group(ner).items():
    correct = n_gold = n_pred = unparsable = wrong_length = 0
    for r in items:
        gold, pred, n_tuples = tags(r["ref"]), tags(r["hyp"]), len(TUPLE.findall(r["hyp"]))
        unparsable += n_tuples == 0
        wrong_length += n_tuples > 0 and n_tuples != len(gold)
        # as tower-eval's align_hyp_ref: cut or pad the prediction to the gold length
        pred = (pred + ["O"] * len(gold))[:len(gold)]
        g, h = chunks(gold), chunks(pred)
        correct, n_gold, n_pred = correct + len(g & h), n_gold + len(g), n_pred + len(h)
        r["ner_correct"], r["ner_gold"], r["ner_pred"] = len(g & h), len(g), len(h)
    P = 100 * correct / n_pred if n_pred else 0.0
    R = 100 * correct / n_gold if n_gold else 0.0
    scores["ner"][lp] = {"n": len(items), "P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
                         "unparsable": unparsable, "wrong_length": wrong_length,
                         "multiline": sum(x["multiline"] for x in items)}

# ---- report ----
if scores["ape"]:
    print("\nAPE (WMT23 test, NLLB-3.3B translations), post-edit vs. the unedited NLLB output:")
    print(f"{'lp':6s} {'n':>5s} {'COMET-22':>9s} {'(nllb)':>7s} {'chrF':>7s} {'(nllb)':>7s} "
          f"{'TER':>7s} {'(nllb)':>7s} {'unch.%':>7s}")
    for lp, s in scores["ape"].items():
        pe, mt = s["post-edit"], s["nllb"]
        print(f"{lp:6s} {s['n']:5d} {pe['COMET-22']:9.2f} {mt['COMET-22']:7.2f} {pe['chrF']:7.2f} {mt['chrF']:7.2f} "
              f"{pe['TER']:7.2f} {mt['TER']:7.2f} {s['unchanged_pct']:7.1f}")
    avg = {k: {m: mean([s[k][m] for s in scores["ape"].values()]) for m in ("COMET-22", "chrF", "TER")}
           for k in ("post-edit", "nllb")}
    scores["ape"]["average"] = avg
    print(f"{'avg':6s} {'':5s} {avg['post-edit']['COMET-22']:9.2f} {avg['nllb']['COMET-22']:7.2f} "
          f"{avg['post-edit']['chrF']:7.2f} {avg['nllb']['chrF']:7.2f} "
          f"{avg['post-edit']['TER']:7.2f} {avg['nllb']['TER']:7.2f}")

if scores["ner"]:
    print("\nNER (MultiCoNER II test), entity-level F1 over the 6 coarse types:")
    print(f"{'lang':6s} {'n':>5s} {'P':>7s} {'R':>7s} {'F1':>7s} {'unparsable':>11s} {'wrong len':>10s}")
    for lp, s in scores["ner"].items():
        print(f"{lp:6s} {s['n']:5d} {s['P']:7.2f} {s['R']:7.2f} {s['F1']:7.2f} {s['unparsable']:11d} "
              f"{s['wrong_length']:10d}")
    scores["ner"]["average"] = {"F1": mean([s["F1"] for s in scores["ner"].values()])}
    print(f"{'avg':6s} {'':5s} {'':7s} {'':7s} {scores['ner']['average']['F1']:7.2f}")

n_multi = sum(r["multiline"] for r in rows)
print(f"\n{n_multi} outputs had text after a newline (cut off before scoring, as tower-eval does)")

base = a.hyp[:-len(".jsonl")] if a.hyp.endswith(".jsonl") else a.hyp  # results/<run>.tasks
scored_path = base + ".scored.jsonl"
with open(scored_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
scores_path = base + "_scores.json"
with open(scores_path, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2)
print("per-segment scores written to", scored_path)
print("aggregate scores written to", scores_path)
