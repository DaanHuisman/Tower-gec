#!/bin/bash
# Step 4: build the GEC training sets and the evaluation sets. Run on a login node (the MT eval set
# needs internet). Existing GEC training files are never overwritten; run setup/verify.py afterwards.
#
#   $TG_DATA/gec_it/{train,dev}-gec_it_merlin.parquet   651 train / 81 dev Italian essays (MultiGEC)
#   $TG_DATA/gec_zh/train-gec_zh_mucgec.parquet         2610 Chinese pairs (MuCGEC, token-matched sample)
#   eval/data/gec_eval.jsonl                            81 Italian essays + 1137 Chinese sentences
#   eval/data/eval_data.jsonl                           FLORES-200 + WMT23 + TICO-19 (MT)
#   eval/data/tasks_eval.jsonl                          APE (WMT23) + NER (MultiCoNER II), from TowerEval-Data
#   eval/data/gec_paper_eval.jsonl + gec_paper_ref/     the Tower paper's GEC sets (en/de/es), from TowerEval-Data
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./config.sh
PY="$TG_ENV/bin/python"

"$PY" data/build_gec_it.py
"$PY" data/build_gec_zh.py

cd eval
"$PY" prepare_gec_data.py
if [ -e data/eval_data.jsonl ]; then
    echo "[skip] eval/data/eval_data.jsonl already exists"
else
    "$PY" prepare_data.py
fi
if [ -e data/tasks_eval.jsonl ] && [ -e data/gec_paper_eval.jsonl ]; then
    echo "[skip] eval/data/tasks_eval.jsonl and gec_paper_eval.jsonl already exist"
else
    "$PY" prepare_tasks_data.py
fi
