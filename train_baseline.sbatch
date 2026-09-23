#!/bin/bash
#SBATCH --job-name=tower-baseline
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/scur0531/tower/logs/%x_%j.out

# Baseline: full fine-tune TowerBase-7B on all of TowerBlocks (all tasks), paper setup (4 epochs).
# For ablations, pass e.g.  TASKS=machine_translation,automatic_post_edition  sbatch train_baseline.sbatch
# Quick smoke test:         MAX_SAMPLES=2000 EPOCHS=1 sbatch train_baseline.sbatch
source ~/tower-env/bin/activate
export HF_HOME=/scratch-shared/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

OUT=/scratch-shared/$USER/runs/${RUN_NAME:-baseline_all}
mkdir -p "$OUT"
cd ~/tower

torchrun --standalone --nproc_per_node=4 train.py \
    --out "$OUT" \
    --epochs "${EPOCHS:-4}" \
    --max_samples "${MAX_SAMPLES:-0}" \
    --tasks "${TASKS:-}"
