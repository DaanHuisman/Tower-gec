#!/bin/bash
#SBATCH --job-name=tower-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

# Evaluate a model. Submit through ./submit.sh eval <run_name|official> [mt|gec|mt,gec].
#   SUITE=mt   FLORES-200 + WMT23 + TICO-19 -> COMET-22 + chrF          (~1.5 h)
#   SUITE=gec  Italian MultiGEC dev + Chinese MuCGEC dev -> GLEU / P / R / F0.5  (~15 min)
# Outputs in results/: <RUN_NAME>.summary.txt (MT), <RUN_NAME>.gec_summary.txt + .gec_scores.json (GEC).
set -e
cd "$SLURM_SUBMIT_DIR"
source ./config.sh
if [ -z "$MODEL" ]; then echo "set MODEL (use ./submit.sh)"; exit 1; fi
source "$TG_ENV/bin/activate"

RUN_NAME=${RUN_NAME:-$(basename "$MODEL")}
SUITE=${SUITE:-mt,gec}
OUT_DIR="$TG_ROOT/results"
mkdir -p "$OUT_DIR"

cd "$TG_ROOT/eval"

if [[ ",$SUITE," == *",mt,"* ]]; then
    python generate.py \
        --model "$MODEL" \
        --out "$OUT_DIR/${RUN_NAME}.jsonl" \
        --batch_size "${BATCH_SIZE:-16}" \
        --testsets "${TESTSETS:-}"

    python score.py --hyp "$OUT_DIR/${RUN_NAME}.jsonl" | tee "$OUT_DIR/${RUN_NAME}.summary.txt"
fi

if [[ ",$SUITE," == *",gec,"* ]]; then
    # Longest reference is ~500 tokens (Italian essays), so allow more than the MT default of 512.
    python generate.py \
        --model "$MODEL" \
        --data data/gec_eval.jsonl \
        --out "$OUT_DIR/${RUN_NAME}.gec.jsonl" \
        --batch_size "${BATCH_SIZE:-16}" \
        --max_new_tokens 1024

    # Official scorers (ChERRANT for Chinese, MultiGEC-2025 GLEU+ERRANT for Italian) run in their own
    # venv (setup/3_scorers.sh). CPU-only, ~10 min.
    "$TG_SCORERS/venv/bin/python" score_gec.py --hyp "$OUT_DIR/${RUN_NAME}.gec.jsonl" \
        | tee "$OUT_DIR/${RUN_NAME}.gec_summary.txt"
fi
