#!/bin/bash
#SBATCH --job-name=tower-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

# Evaluate a model. Submit through ./submit.sh eval <run_name|official> [mt|gec|tasks|gec_paper|mt,gec].
#   SUITE=mt     FLORES-200 + WMT23 + TICO-19 -> COMET-22 + chrF          (~1.5 h)
#   SUITE=gec    Italian MultiGEC dev + Chinese MuCGEC dev -> GLEU / P / R / F0.5  (~15 min)
#   SUITE=tasks  Tower's other tasks: APE (WMT23, NLLB-3.3B output) -> COMET-22 / chrF / TER,
#                NER (MultiCoNER II) -> entity F1   (~1.2 h)
#   SUITE=gec_paper  the Tower paper's GEC sets: English CoNLL-2014, German, Spanish (5-shot)
#                    -> edit rate + ERRANT   (~30 min)
# Outputs in results/: <RUN_NAME>.summary.txt (MT), <RUN_NAME>.gec_summary.txt + .gec_scores.json (GEC),
# <RUN_NAME>.tasks_summary.txt + .tasks_scores.json (APE + NER),
# <RUN_NAME>.gec_paper_summary.txt + .gec_paper_scores.json (the paper's GEC sets).
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

if [[ ",$SUITE," == *",tasks,"* ]]; then
    # Same generation limit as tower-eval (1024 tokens); NER answers list every token with its tag.
    python generate.py \
        --model "$MODEL" \
        --data data/tasks_eval.jsonl \
        --out "$OUT_DIR/${RUN_NAME}.tasks.jsonl" \
        --batch_size "${BATCH_SIZE:-16}" \
        --max_new_tokens 1024

    python score_tasks.py --hyp "$OUT_DIR/${RUN_NAME}.tasks.jsonl" | tee "$OUT_DIR/${RUN_NAME}.tasks_summary.txt"
fi

if [[ ",$SUITE," == *",gec_paper,"* ]]; then
    python generate.py \
        --model "$MODEL" \
        --data data/gec_paper_eval.jsonl \
        --out "$OUT_DIR/${RUN_NAME}.gec_paper.jsonl" \
        --batch_size "${BATCH_SIZE:-16}" \
        --max_new_tokens 1024

    # Official ERRANT (English only, as in tower-eval) + TER, in their own venv (setup/3b_errant.sh). CPU, ~2 min.
    "$TG_SCORERS/errant-venv/bin/python" score_gec_paper.py --hyp "$OUT_DIR/${RUN_NAME}.gec_paper.jsonl" \
        | tee "$OUT_DIR/${RUN_NAME}.gec_paper_summary.txt"
fi
