#!/bin/bash
#SBATCH --job-name=tower-gec
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00

# Continue TowerInstruct-7B on a fixed TowerBlocks replay subset (+ optional GEC data).
# Submit through ./submit.sh train <experiment>, which sets RUN_NAME, EXTRA_DATA etc. and the log path.
# The log prints "replay subset: ... md5=..." -- it must be identical across runs (beea7a1bfd76 for 30k).
set -e
cd "$SLURM_SUBMIT_DIR"
source ./config.sh
if [ -z "$RUN_NAME" ]; then echo "set RUN_NAME (use ./submit.sh)"; exit 1; fi
source "$TG_ENV/bin/activate"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

OUT="$TG_RUNS/$RUN_NAME"
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=4 train.py \
    --model "${MODEL:-$TG_MODELS/TowerInstruct-7B}" \
    --data "$TG_DATA/data" \
    --out "$OUT" \
    --replay_samples "${REPLAY:-30000}" \
    --extra_data ${EXTRA_DATA:-} \
    --epochs "${EPOCHS:-2}" \
    --lr "${LR:-2e-6}" \
    --warmup_steps "${WARMUP:-20}" \
    --seed "${SEED:-42}"
