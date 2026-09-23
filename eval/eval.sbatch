#!/bin/bash
#SBATCH --job-name=tower-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/scur0531/tower/eval/logs/%x_%j.out

# Evaluate a Tower checkpoint on FLORES-200 + WMT23 + TICO-19 with COMET-22.
# eval/data/eval_data.jsonl must already exist (build it on the login node first:
#   python eval/prepare_data.py    -- needs internet, run once).
#
# This cluster does not auto-export inline `VAR=val sbatch ...` assignments, so
# MODEL/RUN_NAME/TESTSETS/BATCH_SIZE must be listed explicitly in --export.
#
# Usage:
#   MODEL=~/models/TowerInstruct-7B RUN_NAME=towerinstruct_7b_official \
#     sbatch --export=ALL,MODEL,RUN_NAME eval.sbatch
#   MODEL=/scratch-shared/$USER/runs/baseline_all/final RUN_NAME=our_baseline \
#     sbatch --export=ALL,MODEL,RUN_NAME eval.sbatch
# Smoke test on a couple of pairs first:
#   MODEL=~/models/TowerInstruct-7B RUN_NAME=smoke TESTSETS=tico19 \
#     sbatch --export=ALL,MODEL,RUN_NAME,TESTSETS eval.sbatch

set -e
source ~/tower-env/bin/activate
export HF_HOME=/scratch-shared/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false

if [ -z "$MODEL" ]; then echo "set MODEL=<path or HF id>"; exit 1; fi
RUN_NAME=${RUN_NAME:-$(basename "$MODEL")}
OUT_DIR=~/tower/eval/results
mkdir -p "$OUT_DIR"
HYP="$OUT_DIR/${RUN_NAME}.jsonl"

cd ~/tower/eval

python generate.py \
    --model "$MODEL" \
    --out "$HYP" \
    --batch_size "${BATCH_SIZE:-16}" \
    --testsets "${TESTSETS:-}"

python score.py --hyp "$HYP" | tee "$OUT_DIR/${RUN_NAME}.summary.txt"
