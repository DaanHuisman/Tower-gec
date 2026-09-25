"""Build Italian GEC data (MultiGEC-2025, MERLIN) in TowerBlocks format.

Reads $TG_DATA/multigec-2025-participants-1/italian/Merlin (MultiGEC-2025 release, request access first).
Writes into $TG_DATA/gec_it/ (kept out of data/ so train.py's default glob never picks it up):
  train-gec_it_merlin.parquet  651 essays, for training
  dev-gec_it_merlin.parquet     81 essays with essay_id + reference, held out for evaluation
MultiGEC-2025 ships no references for the test split, so dev is the only labelled eval set.
"""
import os, re, sys
import pandas as pd

DATA = os.environ.get("TG_DATA", os.path.expanduser("~/data/TowerBlocks"))
SRC = os.path.join(DATA, "multigec-2025-participants-1/italian/Merlin")
OUT = os.path.join(DATA, "gec_it")
PROMPT = ("GEC is a task designed to correct grammatical, orthographic, and other minor errors in a piece "
          "of text, making only the edits necessary to fix mistakes while preserving the original meaning.\n"
          "Original (Italian): {src}\nCorrected: ")


def read_md(path):
    """Return {essay_id: text} from a MultiGEC .md file."""
    parts = re.split(r"^### essay_id = (\S+)\s*$", open(path, encoding="utf8").read(), flags=re.M)
    return {eid: text.strip() for eid, text in zip(parts[1::2], parts[2::2])}


def build(split):
    orig = read_md(f"{SRC}/it-merlin-orig-{split}.md")
    ref = read_md(f"{SRC}/it-merlin-ref1-{split}.md")
    assert orig.keys() == ref.keys(), f"{split}: orig/ref essay ids differ"
    return pd.DataFrame([{
        "conversations": [{"from": "human", "value": PROMPT.format(src=orig[eid])},
                          {"from": "gpt", "value": ref[eid]}],
        "lang": "it", "split": split, "dataset": "merlin_it_gec",
        "task": "grammatical_error_correction", "essay_id": eid, "source": orig[eid], "reference": ref[eid],
    } for eid in orig])


if __name__ == "__main__":
    targets = [f"{OUT}/train-gec_it_merlin.parquet", f"{OUT}/dev-gec_it_merlin.parquet"]
    if all(os.path.exists(t) for t in targets):
        print(f"{OUT} already built, not overwriting (setup/verify.py checks its contents)")
        sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    train, dev = build("train"), build("dev")
    assert not set(train.essay_id) & set(dev.essay_id), "train/dev essay overlap"
    assert not set(train.source) & set(dev.source), "train/dev source-text overlap"
    # train.py concatenates parquet files, so the training file keeps exactly the TowerBlocks columns
    train.drop(columns=["essay_id", "source", "reference"]).to_parquet(f"{OUT}/train-gec_it_merlin.parquet")
    dev.to_parquet(f"{OUT}/dev-gec_it_merlin.parquet")
    print(f"train={len(train)} dev={len(dev)} -> {OUT}")
