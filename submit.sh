#!/bin/bash
# Submit experiments to Slurm. Run from anywhere; logs go to logs/, results to results/.
#
#   ./submit.sh train <experiment> [sbatch args]              train one model (see EXPERIMENTS below)
#   ./submit.sh eval <run_name|official> [suite] [sbatch args] evaluate a trained run (or the official
#                                                             TowerInstruct-7B); suite = mt | gec | tasks | gec_paper | mt,gec (default)
#   ./submit.sh all                                 train all four experiments, each followed by its eval
#   ./submit.sh tasks [sbatch args]                 APE + NER evals of official, gec_control, gec_it, gec_zh
#   ./submit.sh gec_paper [sbatch args]             the Tower paper's GEC sets (en/de/es) for the same four
#   ./submit.sh baseline                            optional: full TowerBase -> TowerInstruct reproduction (days)
#
# Hyperparameters can be overridden per call: REPLAY=20000 EPOCHS=1 LR=1e-6 ./submit.sh train it
# Repeat a run with another training seed (same replay subset, different data order):
#   SEED=1 ./submit.sh train it    -> run gec_it_s1 (the default seed 42 gives gec_it)
# Extra sbatch args are passed through, e.g. start an eval once training succeeded:
#   ./submit.sh eval gec_it gec --dependency=afterok:<training job id>
#
# EXPERIMENTS (all start from TowerInstruct-7B and share the same 30k TowerBlocks replay subset):
#   control  replay only                         -> run gec_control
#   it       replay + Italian GEC (651 essays)   -> run gec_it
#   zh       replay + Chinese GEC (2610 pairs)   -> run gec_zh
#   it_zh    replay + Italian + Chinese GEC      -> run gec_it_zh
#   smoke    tiny end-to-end check (~3 min)      -> run gec_smoke
set -e
shopt -s inherit_errexit  # also stop on errors inside $(...)
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh
mkdir -p logs results

# This cluster does not reliably pass inline `VAR=val sbatch` assignments to the job,
# so every variable the job needs is exported here and named in --export.
submit() {  # submit <sbatch file> <VAR names to export...> [-- extra sbatch args]
    local script=$1; shift
    local vars="ALL" extra=()
    while [ $# -gt 0 ]; do
        if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
        vars="$vars,$1"; shift
    done
    sbatch --parsable --output="$TG_ROOT/logs/%x_%j.out" --export="$vars" "${extra[@]}" "$script"
}

experiment_data() {  # prints EXTRA_DATA for an experiment
    case $1 in
        control) echo "" ;;
        it)      echo "$TG_DATA/gec_it" ;;
        zh)      echo "$TG_DATA/gec_zh" ;;
        it_zh)   echo "$TG_DATA/gec_it $TG_DATA/gec_zh" ;;
        smoke)   echo "$TG_DATA/gec_it:3 $TG_DATA/gec_zh" ;;
        *) echo "unknown experiment '$1' (control|it|zh|it_zh|smoke)" >&2; exit 1 ;;
    esac
}

train() {  # train <experiment> [extra sbatch args...]; prints the job id
    local exp=$1; shift
    export SEED=${SEED:-42}
    export RUN_NAME="gec_$exp"; [ "$SEED" = 42 ] || RUN_NAME+="_s$SEED"
    export EXTRA_DATA; EXTRA_DATA=$(experiment_data "$exp") || exit 1
    local extra=("$@")
    if [ "$exp" = smoke ]; then
        export REPLAY=${REPLAY:-2000} EPOCHS=${EPOCHS:-0.2}
        extra+=(--time=01:00:00)
    fi
    export REPLAY=${REPLAY:-30000} EPOCHS=${EPOCHS:-2} LR=${LR:-2e-6} WARMUP=${WARMUP:-20}
    submit slurm/train_gec.sbatch RUN_NAME EXTRA_DATA REPLAY EPOCHS LR WARMUP SEED -- "${extra[@]}"
}

evaluate_finished_runs() {  # evaluate_finished_runs <suite> [extra sbatch args...]
    # the baseline and three of the runs; gec_it_zh is not included
    local suite=$1 run job; shift
    for run in official gec_control gec_it gec_zh; do
        job=$(evaluate "$run" "$suite" "$@"); echo "submitted $suite eval of $run: job $job (log: logs/tower-eval_$job.out)"
    done
    echo "when all four are done: \$TG_ENV/bin/python eval/compare_tasks.py"
}

evaluate() {  # evaluate <run_name|official> [suite] [extra sbatch args...]; prints the job id
    local run=$1 suite=${2:-mt,gec}; shift; [ $# -gt 0 ] && shift
    if [ "$run" = official ]; then
        export MODEL="$TG_MODELS/TowerInstruct-7B" RUN_NAME=towerinstruct_7b_official
    else
        export MODEL="$TG_RUNS/$run/final" RUN_NAME=$run
    fi
    if [ ! -e "$MODEL/config.json" ] && [[ " $* " != *"--dependency"* ]]; then
        echo "no trained model at $MODEL (is the run name right, and has training finished?)" >&2; exit 1
    fi
    export SUITE=$suite
    local extra=("$@")
    [ "$suite" = gec ] && extra+=(--time=01:00:00)
    [ "$suite" = tasks ] && extra+=(--time=03:00:00)
    [ "$suite" = gec_paper ] && extra+=(--time=01:30:00)
    submit slurm/eval.sbatch MODEL RUN_NAME SUITE -- "${extra[@]}"
}

case $1 in
    train)
        [ -n "$2" ] || { sed -n 2,21p "$0"; exit 1; }
        experiment_data "$2" > /dev/null  # rejects unknown experiment names before submitting
        job=$(train "$2" "${@:3}"); echo "submitted training gec_$2 (seed ${SEED:-42}): job $job (log: logs/tower-gec_$job.out)" ;;
    eval)
        [ -n "$2" ] || { sed -n 2,21p "$0"; exit 1; }
        job=$(evaluate "$2" "${3:-mt,gec}" "${@:4}"); echo "submitted eval of $2: job $job (log: logs/tower-eval_$job.out)" ;;
    all)
        for exp in control it zh it_zh; do
            t=$(train "$exp")
            e=$(evaluate "gec_$exp" mt,gec --dependency=afterok:$t)
            echo "gec_$exp: train job $t, eval job $e (starts when training succeeds)"
        done ;;
    tasks|gec_paper)
        evaluate_finished_runs "$1" "${@:2}" ;;
    baseline)
        export RUN_NAME=baseline_all
        job=$(submit slurm/train_baseline.sbatch RUN_NAME); echo "submitted baseline reproduction: job $job" ;;
    *)
        sed -n 2,21p "$0"; exit 1 ;;
esac
