#!/bin/bash
# Step 2: download models and data at the exact revisions used. Run on a login node (needs internet).
# Anything that already exists is skipped, never overwritten.
#
#   TowerInstruct-7B-v0.1   -> $TG_MODELS/TowerInstruct-7B   (26 GB, the model all experiments start from)
#   TowerBase-7B-v0.1       -> $TG_MODELS/TowerBase-7B       (tokenizer only; WITH_TOWERBASE=1 for the
#                                                            27 GB weights, only needed for ./submit.sh baseline)
#   TowerBlocks-v0.1        -> $TG_DATA/data/                (the replay data)
#   MuCGEC train + dev      -> $TG_DATA/mucgec/              (Chinese GEC)
#   COMET-22                -> $HF_HOME                      (MT metric)
#
# NOT downloadable by script: MultiGEC-2025 (Italian GEC). Its licence forbids redistribution, so every
# group member must request access themselves -- see README.md, section "MultiGEC access".
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./config.sh
HF="$TG_ENV/bin/hf"
PY="$TG_ENV/bin/python"
mkdir -p "$TG_MODELS" "$TG_DATA" "$HF_HOME"

# ---- models (pinned commits) ----
if [ -e "$TG_MODELS/TowerInstruct-7B/model-00006-of-00006.safetensors" ]; then
    echo "[skip] TowerInstruct-7B already in $TG_MODELS"
else
    "$HF" download Unbabel/TowerInstruct-7B-v0.1 --revision cb2e455b98c8e887e55729f37773fba95a578325 \
        --local-dir "$TG_MODELS/TowerInstruct-7B"
fi

if [ "${WITH_TOWERBASE:-0}" = 1 ]; then
    if [ -e "$TG_MODELS/TowerBase-7B/model.safetensors" ]; then
        echo "[skip] TowerBase-7B weights already in $TG_MODELS"
    else
        "$HF" download Unbabel/TowerBase-7B-v0.1 --revision bd59c10d76007e960368acfc15bebc201c112b1b \
            --exclude "pytorch_model.bin" --local-dir "$TG_MODELS/TowerBase-7B"
    fi
elif [ -e "$TG_MODELS/TowerBase-7B/tokenizer.json" ]; then
    echo "[skip] TowerBase-7B tokenizer already in $TG_MODELS"
else
    # Only the tokenizer is needed: data/build_gec_zh.py counts tokens with it.
    "$HF" download Unbabel/TowerBase-7B-v0.1 --revision bd59c10d76007e960368acfc15bebc201c112b1b \
        --include "tokenizer*" "special_tokens_map.json" "config.json" "generation_config.json" \
        --local-dir "$TG_MODELS/TowerBase-7B"
fi

# ---- TowerBlocks ----
if [ "$(ls "$TG_DATA"/data/train-0000?-of-00004.parquet 2>/dev/null | wc -l)" = 4 ]; then
    echo "[skip] TowerBlocks already in $TG_DATA/data"
else
    "$HF" download Unbabel/TowerBlocks-v0.1 --repo-type dataset \
        --revision 372dc8d7378ee47013a9f4c21cd656cf516bb051 --local-dir "$TG_DATA"
fi

# ---- MuCGEC ----
MU="$TG_DATA/mucgec"
if [ -e "$MU/MuCGEC_exp_data/train/train.para" ]; then
    echo "[skip] MuCGEC training data already in $MU"
else
    # Training data (Lang-8 + HSK) + dev M2, from the Google Drive link in the MuCGEC README.
    # If gdown hits a Drive quota error, download the file in a browser from
    # https://drive.google.com/uc?id=1l0A50z7fMXjQT3y2ct7TQsEHqOwlvg0_ , save it as $MU/mucgec_exp_data.zip
    # and rerun this script.
    mkdir -p "$MU"
    GDOWN="$TG_ENV/bin/gdown"  # installed by setup/1_env.sh; fall back to one on PATH for older envs
    [ -x "$GDOWN" ] || GDOWN=$(command -v gdown || echo "$HOME/.local/bin/gdown")
    [ -x "$GDOWN" ] || { echo "gdown not found: pip install gdown==5.2.2"; exit 1; }
    [ -e "$MU/mucgec_exp_data.zip" ] || "$GDOWN" "https://drive.google.com/uc?id=1l0A50z7fMXjQT3y2ct7TQsEHqOwlvg0_" -O "$MU/mucgec_exp_data.zip"
    (cd "$MU" && unzip -q -o mucgec_exp_data.zip -x '__MACOSX/*' '*.DS_Store' && rm -f mucgec_exp_data.zip)
fi
if [ -e "$MU/eval_repo_data/MuCGEC_dev.txt" ]; then
    echo "[skip] MuCGEC dev set already in $MU"
else
    mkdir -p "$MU/eval_repo_data"
    curl -fsSL -o "$MU/eval_repo_data/MuCGEC_dev.txt" \
        https://raw.githubusercontent.com/HillZhang1999/MuCGEC/39221934ee4ccf07801879136a75dd7e11319977/data/MuCGEC/MuCGEC_dev.txt
fi

# ---- COMET-22 (MT metric), cached so evaluation jobs do not depend on the network ----
if "$PY" -c "from comet import download_model, load_from_checkpoint; load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))" > /dev/null 2>&1; then
    echo "[ok] COMET-22 cached in $HF_HOME"
else
    echo "[warn] could not cache COMET-22; evaluation jobs will try to download it themselves"
fi

# ---- MultiGEC (manual) ----
IT="$TG_DATA/multigec-2025-participants-1/italian/Merlin"
if [ -e "$IT/it-merlin-orig-train.md" ]; then
    echo "[ok] MultiGEC Italian data found in $IT"
else
    echo
    echo "[MISSING] MultiGEC-2025 is not in $TG_DATA/multigec-2025-participants-1"
    echo "  Request access at https://lt3.ugent.be/resources/multigec-dataset, then unpack the release so that"
    echo "  $IT/it-merlin-orig-train.md exists. See README.md."
fi
