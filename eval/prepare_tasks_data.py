"""Build the evaluation sets for Tower's other tasks, from TowerEval-Data-v0.1 (the data the Tower paper
evaluates with). Run this on the LOGIN NODE (needs internet). The prompts are TowerEval's, unchanged.
  eval/data/tasks_eval.jsonl       0-shot TowerInstruct prompts
    APE  automatic post-editing of NLLB-3.3B translations of WMT23, 6 language pairs (8,953 segments)
    NER  named-entity recognition, MultiCoNER II test, 1,000 sentences each for en/de/es/fr/it/pt/zh
  eval/data/gec_paper_eval.jsonl   5-shot TowerInstruct prompts (the paper holds GEC out of training)
    GEC  CoNLL-2014 (en, 1,312), Falko-MERLIN from mlconvgec2018 (de, 2,337), COWSL2H (es, 821)
  eval/data/gec_paper_ref/<testset>.<lang>.m2   ERRANT reference annotations of the GEC sets
Fields: task, testset, lp, src, ref, prompt (+ mt for APE: the NLLB translation to post-edit;
+ tok_src for GEC: the tokenised source ERRANT aligns against). For NER and GEC, lp is the language;
for NER, src is the token list and ref the gold [("token", "tag"), ...] list.
"""
import json
import os
import tarfile
from collections import Counter

from huggingface_hub import hf_hub_download

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

REPO, REVISION = "Unbabel/TowerEval-Data-v0.1", "41fadea4717dde62560831bf650f2f969b0a051c"
TASKS = {  # task -> (testset, subtasks), as in tower-eval's configs/tower_paper/tower_instruct_0_shot.yaml
    "ape": ("nllb_3b_wmt23", ["en-de", "de-en", "en-ru", "ru-en", "en-zh", "zh-en"]),
    "ner": ("multiconer2023", ["en", "de", "es", "fr", "it", "pt", "zh"]),
}
GEC = ["conll14.en", "fm.de", "cowsl2h.es"]  # configs/tower_paper/tower_instruct_5_shot.yaml

tar = tarfile.open(hf_hub_download(REPO, "data.tar.gz", repo_type="dataset", revision=REVISION))


def lines(member):
    # split on "\n" only: str.splitlines() would also split on characters like U+2028 inside the text
    return tar.extractfile(member).read().decode("utf-8").rstrip("\n").split("\n")


def tests_and_prompts(task, name, setting):
    tests = [json.loads(l) for l in lines(f"data/raw_data/{task}/{name}/test.jsonl")]
    # instructions.txt holds one prompt per line with newlines escaped; tower-eval unescapes them
    prompts = [l.replace("\\n", "\n") for l in lines(f"data/instructions/{setting}/{task}/{name}/instructions.txt")]
    assert len(tests) == len(prompts), f"{name}: {len(tests)} test rows but {len(prompts)} prompts"
    return zip(tests, prompts)


def write(rows, name):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    c = Counter((r["task"], r["lp"]) for r in rows)
    print(f"wrote {len(rows)} examples to {path}")
    for k in sorted(c):
        print(f"  {k[0]:4s} {k[1]:6s} {c[k]}")


os.makedirs(os.path.join(DATA_DIR, "gec_paper_ref"), exist_ok=True)

rows = []
for task, (testset, subtasks) in TASKS.items():
    for sub in subtasks:
        name = f"{testset}.{sub}"
        for t, prompt in tests_and_prompts(task, name, "0_shot_tower_instruct"):
            if task == "ape":
                assert t["mt"] in prompt, f"{name}: prompt does not match its test row"
                rows.append({"task": task, "testset": testset, "lp": sub, "src": t["src"], "mt": t["mt"],
                             "ref": t["ref"], "prompt": prompt})
            else:
                assert t["input"] in prompt, f"{name}: prompt does not match its test row"
                rows.append({"task": task, "testset": testset, "lp": sub, "src": t["input"],
                             "ref": t["answer"], "prompt": prompt})

write(rows, "tasks_eval.jsonl")

rows = []
for name in GEC:
    testset, lang = name.split(".")
    m2 = tar.extractfile(f"data/raw_data/gec/{name}/test_corpus.m2").read().decode("utf-8")
    m2_sources = [l[2:] for l in m2.split("\n") if l.startswith("S ")]
    pairs = list(tests_and_prompts("gec", name, "5_shot_tower_instruct"))
    assert len(m2_sources) == len(pairs), f"{name}: {len(m2_sources)} M2 sentences but {len(pairs)} test rows"
    for (t, prompt), m2_src in zip(pairs, m2_sources):
        assert f"Source: {t['src']}\nReference: <|im_end|>" in prompt, f"{name}: prompt does not match its test row"
        assert t["tok_src"].split() == m2_src.split(), f"{name}: tok_src does not match the M2 source"
        rows.append({"task": "gec", "testset": testset, "lp": lang, "src": t["src"], "tok_src": t["tok_src"],
                     "ref": t["ref"], "prompt": prompt})
    with open(os.path.join(DATA_DIR, "gec_paper_ref", f"{name}.m2"), "w", encoding="utf-8") as f:
        f.write(m2)
write(rows, "gec_paper_eval.jsonl")
