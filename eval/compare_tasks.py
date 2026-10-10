"""Put the scores on Tower's other tasks of several runs side by side: APE + NER
(results/<run>.tasks_scores.json) and the Tower paper's GEC sets (results/<run>.gec_paper_scores.json).
A section is printed for the runs whose scores exist.

    $TG_ENV/bin/python eval/compare_tasks.py                       # official, control, it, zh
    $TG_ENV/bin/python eval/compare_tasks.py gec_control gec_it    # any runs
"""
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
runs = sys.argv[1:] or ["towerinstruct_7b_official", "gec_control", "gec_it", "gec_zh"]
label = {"towerinstruct_7b_official": "official", "gec_control": "control", "gec_it": "it", "gec_zh": "zh"}


def load(suffix, command):
    scores = {}
    for run in runs:
        path = os.path.join(RESULTS, f"{run}.{suffix}")
        if os.path.exists(path):
            scores[run] = json.load(open(path))
        else:
            print(f"[missing] {path} (has ./submit.sh {command} finished for {run}?)")
    return scores


def row(name, values):
    print(f"{name:12s}" + "".join(f"{v:9.2f}" for v in values))


def header(scores, extra=()):
    print(f"{'':12s}" + "".join(f"{label.get(r, r):>9s}" for r in scores) + "".join(f"{n:>9s}" for n in extra))


tasks = load("tasks_scores.json", "tasks")
gec = load("gec_paper_scores.json", "gec_paper")
if not tasks and not gec:
    sys.exit(1)

if tasks:
    first = next(iter(tasks.values()))
    print("\nAPE (WMT23, post-editing NLLB-3.3B output), average over 6 language pairs")
    header(tasks, ["NLLB"])
    for metric in ("COMET-22", "chrF", "TER"):
        nllb = first["ape"]["average"]["nllb"][metric]
        row(metric, [s["ape"]["average"]["post-edit"][metric] for s in tasks.values()] + [nllb])

    print("\nAPE COMET-22 per language pair")
    header(tasks, ["NLLB"])
    for lp in (k for k in first["ape"] if k != "average"):
        row(lp, [s["ape"][lp]["post-edit"]["COMET-22"] for s in tasks.values()] + [first["ape"][lp]["nllb"]["COMET-22"]])

    print("\nNER (MultiCoNER II) F1 per language")
    header(tasks)
    for lang in (k for k in first["ner"] if k != "average"):
        row(lang, [s["ner"][lang]["F1"] for s in tasks.values()])
    row("average", [s["ner"]["average"]["F1"] for s in tasks.values()])

if gec:
    first = next(iter(gec.values()))
    langs = [k for k in first if k != "average"]
    print("\nGEC, Tower paper's test sets (5-shot): edit rate, lower is better")
    header(gec, ["no edits"])
    for lang in langs:
        row(lang, [s[lang]["edit_rate"] for s in gec.values()] + [first[lang]["edit_rate_no_edits"]])
    row("average", [s["average"]["edit_rate"] for s in gec.values()]
        + [sum(first[l]["edit_rate_no_edits"] for l in langs) / len(langs)])

    print("\nGEC, Tower paper's test sets (5-shot): ERRANT F0.5")
    header(gec)
    for lang in langs:
        row(lang, [s[lang]["errant"]["F0.5"] for s in gec.values()])
    row("average", [s["average"]["errant_F0.5"] for s in gec.values()])
