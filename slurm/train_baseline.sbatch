#!/bin/bash
#SBATCH --job-name=tower-baseline
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --time=4-00:00:00

# OPTIONAL, not part of the main experiment: reproduce TowerInstruct by fully fine-tuning TowerBase-7B on
# all of TowerBlocks (paper setup, 4 epochs; days of compute). Needs the full TowerBase weights
# (WITH_TOWERBASE=1 bash setup/2_download.sh). Submit with ./submit.sh baseline.
set -e
cd "$SLURM_SUBMIT_DIR"
source ./config.sh
source "$TG_ENV/bin/activate"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

OUT="$TG_RUNS/${RUN_NAME:-baseline_all}"
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=4 train.py \
    --model "$TG_MODELS/TowerBase-7B" \
    --data "$TG_DATA/data" \
    --out "$OUT" \
    --epochs "${EPOCHS:-4}" \
    --max_samples "${MAX_SAMPLES:-0}" \
    --tasks "${TASKS:-}"
