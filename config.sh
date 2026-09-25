# Paths used by every script in this repo. Sourced by setup/*.sh, submit.sh and slurm/*.sbatch.
# Defaults work on Snellius. To use other locations, export the variable before running anything
# (e.g. in ~/.bashrc) -- do not hardcode personal paths in the scripts themselves.

TG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TG_ROOT

export TG_DATA="${TG_DATA:-$HOME/data/TowerBlocks}"      # TowerBlocks, MuCGEC, MultiGEC, built GEC sets (~1.2 GB)
export TG_MODELS="${TG_MODELS:-$HOME/models}"            # TowerInstruct-7B/ (26 GB), TowerBase-7B/
export TG_ENV="${TG_ENV:-$HOME/tower-env}"               # Python 3.9 venv for training + evaluation
export TG_SCORERS="${TG_SCORERS:-$HOME/gec-scorers}"     # ChERRANT + MultiGEC scorers and their Python 3.12 venv
export TG_RUNS="${TG_RUNS:-/scratch-shared/$USER/runs}"  # trained models, ~51 GB per run (scratch is purged, copy what you keep)
export HF_HOME="${HF_HOME:-/scratch-shared/$USER/hf_cache}"  # Hugging Face cache (COMET, FLORES)

# Snellius module that provides Python 3.12 for the scorer venv (only used by setup/3_scorers.sh).
export TG_PY312_MODULES="${TG_PY312_MODULES:-2024 Python/3.12.3-GCCcore-13.3.0}"

export TOKENIZERS_PARALLELISM=false
