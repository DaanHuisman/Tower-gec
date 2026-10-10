# Adding GEC to Tower's training mix

This repository contains the code and scores for our paper on adding grammatical error correction (GEC)
data to the training mix of a translation LLM. We continue training
[TowerInstruct-7B](https://huggingface.co/Unbabel/TowerInstruct-7B-v0.1) with Italian and/or Chinese GEC
data, then measure whether GEC improves and what that costs on translation and Tower's other tasks.

The Tower authors suggest that TowerInstruct's average GEC performance comes from the absence of GEC data
in its training data (TowerBlocks). These experiments test that.

## What is in this repository

- **Code** for every step:

  | Step | Files |
  |---|---|
  | Build the GEC training and evaluation sets | `data/`, `eval/prepare_*.py` |
  | Fine-tune | `train.py` (full fine-tune, loss on assistant turns only) |
  | Generate | `eval/generate.py` (greedy decoding, Tower chat template) |
  | Score | `eval/score*.py` (official scorers; see [Implementation notes](#implementation-notes)) |
  | Significance tests | `eval/significance.py`, `eval/m2_counts.py` |
  | Run everything on SLURM | `submit.sh`, `slurm/`, `config.sh` |

- **Scores** behind every number in the paper, in `results/` (see [Where the numbers come from](#where-the-numbers-come-from)).
- **Not included:**
  - MultiGEC-2025 data and anything built from it, because its licence forbids redistribution.
  - Raw model outputs (`results/*.jsonl`), because they contain the test sentences and references,
    including MultiGEC.
  - Trained models (about 51 GB each).

The setup scripts download everything else.

## Experiments

Every run continues training the released TowerInstruct-7B on the same **fixed random subset of 30k
TowerBlocks examples** (the "replay" data, so the model keeps practising its existing tasks). The only
difference between runs is the GEC data added on top:

| Run | Training data | Question it answers |
|---|---|---|
| `gec_control` | replay only | What does extra training by itself do? |
| `gec_it` | replay + Italian GEC (651 essays) | GEC gain for a language with little GEC data |
| `gec_zh` | replay + Chinese GEC (2,610 sentence pairs) | GEC gain for a different writing system |
| `gec_it_zh` | replay + both | Can one model learn both? Do they interfere? |
| official TowerInstruct-7B | no extra training | Reference point |

Training settings: 2 epochs, learning rate 2e-6 with cosine decay and 20 warmup steps, batch size 256,
full fine-tune on 4 H100s. Each run was trained once, with seed 42.

## Evaluation

- **GEC:** Italian MultiGEC-2025 dev (81 essays; GLEU and ERRANT P/R/F0.5) and Chinese MuCGEC dev
  (1,137 sentences; char-level ChERRANT P/R/F0.5), both with the official scorers.
- **Translation:** FLORES-200, WMT23 and TICO-19 (COMET-22 and chrF). This is where a cost from GEC
  training would show up.
- **Tower's other tasks:** automatic post-editing of NLLB-3.3B translations of WMT23 (6 language pairs;
  COMET-22, chrF, TER) and named-entity recognition on MultiCoNER II (7 languages; entity F1). Data and
  prompts are TowerEval-Data-v0.1, the set the Tower paper uses, and scoring follows tower-eval.
- **The Tower paper's GEC sets:** English CoNLL-2014, German Falko-MERLIN and Spanish COWSL2H with the
  paper's 5-shot prompts, scored with edit rate (TER against the reference) and official ERRANT. These
  show whether Italian or Chinese GEC training carries over to other languages.

`gec_it_zh` was evaluated on GEC and translation only. It was not run on Tower's other tasks or the
paper's GEC sets, and it is not in the significance tests.

## Where the numbers come from

| File in `results/` | Contents |
|---|---|
| `RESULTS.txt` | GEC and translation results for official, control, it and zh, and how each number was verified |
| `<run>.gec_summary.txt`, `<run>.gec_scores.json` | GEC on the Italian and Chinese dev sets |
| `<run>.summary.txt` | translation: COMET-22 and chrF per test set and language pair |
| `<run>.tasks_summary.txt`, `<run>.tasks_scores.json` | post-editing and NER |
| `<run>.gec_paper_summary.txt`, `<run>.gec_paper_scores.json` | the Tower paper's GEC sets |
| `SIGNIFICANCE.md`, `significance.json` | paired bootstrap tests (1,000 resamples) between the models |
| `PAPER_TESTS.md` | every evaluation in the Tower paper, whether we ran it, and our numbers next to the paper's |
| `reference/` | the official model's GEC and translation scores |

`<run>` is `gec_control`, `gec_it`, `gec_zh`, `gec_it_zh` or `towerinstruct_7b_official`.

GEC on our dev sets, from `<run>.gec_summary.txt`:

| Run | Italian GLEU | Italian P | Italian R | Italian F0.5 | Chinese P | Chinese R | Chinese F0.5 |
|---|---|---|---|---|---|---|---|
| official | 58.82 | 54.95 | 27.47 | 45.79 | 27.24 | 14.57 | 23.20 |
| `gec_control` | 56.54 | 54.66 | 24.39 | 43.79 | 26.04 | 16.79 | 23.45 |
| `gec_it` | 72.76 | 58.96 | 47.92 | 56.36 | 27.72 | 20.90 | 26.02 |
| `gec_zh` | 56.85 | 57.80 | 23.39 | 44.66 | 42.49 | 11.76 | 27.90 |
| `gec_it_zh` | 71.76 | 58.18 | 46.70 | 55.45 | 41.92 | 12.37 | 28.37 |

## Reproducing the results

### Requirements

- A SLURM cluster with H100 GPUs. Per model, training takes about 35 minutes on 4 GPUs. Evaluation on
  1 GPU takes about 1.5 hours for translation and GEC, about 1.5 hours for post-editing and NER, and
  about 30 minutes for the paper's GEC sets.
- About 30 GB of disk for models and data, plus about 51 GB per trained model.
- Access to MultiGEC-2025 (see step 1).

We ran everything on Snellius (SURF). On another SLURM cluster, change:
- `--partition=gpu_h100` in `slurm/*.sbatch`
- `PY=/usr/bin/python3` in `setup/1_env.sh` (any Python 3.9)
- `TG_PY312_MODULES` in [config.sh](config.sh), the environment module that provides Python 3.12 for the scorers
- the default paths in [config.sh](config.sh) (see the table below)

### 1. Request MultiGEC access (it takes time)

The Italian data comes from the MultiGEC-2025 release. Its terms forbid redistribution, so you need to
request access yourself at <https://lt3.ugent.be/resources/multigec-dataset>. The terms are research use
only, no sharing, and no feeding the data to proprietary models that retain it for training.

Once approved, unpack the release so that this file exists:
`$TG_DATA/multigec-2025-participants-1/italian/Merlin/it-merlin-orig-train.md`
(with the default config, `$TG_DATA` is `~/data/TowerBlocks`).

### 2. Install, download, build, verify

```bash
git clone <this repo> tower-gec && cd tower-gec
bash setup/1_env.sh          # Python 3.9 venv with pinned packages               (~10 min)
bash setup/2_download.sh     # TowerInstruct-7B, TowerBlocks, MuCGEC, COMET-22     (~15 min, 28 GB)
bash setup/3_scorers.sh      # ChERRANT + MultiGEC scorers in a Python 3.12 venv    (~10 min)
bash setup/3b_errant.sh      # official ERRANT for the paper's GEC sets, own venv   (~1 min)
bash setup/4_build_data.sh   # GEC training sets + evaluation sets                  (~2 min)
source config.sh && $TG_ENV/bin/python setup/verify.py
```

`verify.py` must end with **"All checks passed."** It checks every downloaded file (md5), the content of
the built GEC sets, the evaluation sets, and the fingerprint of the 30k replay subset (`beea7a1bfd76`).
If all of these match, you have exactly the data we used.

All scripts skip anything that already exists and never overwrite data, so they are safe to rerun.

Default locations are set in [config.sh](config.sh). To use other locations, export the variable before
running anything, e.g. `export TG_MODELS=/data/models`.

| Variable | Default | Contents |
|---|---|---|
| `TG_DATA` | `~/data/TowerBlocks` | TowerBlocks, MuCGEC, MultiGEC, built GEC sets (1.2 GB) |
| `TG_MODELS` | `~/models` | TowerInstruct-7B (26 GB), TowerBase-7B tokenizer |
| `TG_ENV` | `~/tower-env` | training and evaluation venv |
| `TG_SCORERS` | `~/gec-scorers` | GEC scorers and their venvs |
| `TG_RUNS` | `/scratch-shared/$USER/runs` | trained models, about 51 GB each |
| `HF_HOME` | `/scratch-shared/$USER/hf_cache` | Hugging Face cache |

### 3. Check the pipeline (optional, about 20 minutes)

A 3-minute training run, then generation and scoring on the GEC sets:

```bash
./submit.sh train smoke                                     # prints "... job <id> ..."
./submit.sh eval gec_smoke gec --dependency=afterok:<id>    # starts when training succeeds
cat results/gec_smoke.gec_summary.txt                       # when both jobs are done (squeue -u $USER)
```

The smoke model has barely changed, so its GEC scores should be within about ±0.5 of the official model's.

### 4. Run the experiments

```bash
./submit.sh eval official             # the official TowerInstruct-7B
./submit.sh all                       # train control, it, zh and it_zh, each followed by its eval

# or one at a time:
./submit.sh train it                  # control | it | zh | it_zh
./submit.sh eval gec_it               # mt | gec | mt,gec (default)

# Tower's other tasks and the paper's GEC sets, for official, gec_control, gec_it and gec_zh:
./submit.sh tasks
./submit.sh gec_paper
$TG_ENV/bin/python eval/compare_tasks.py   # side-by-side tables once the jobs are done

# paired bootstrap significance tests (CPU, ~1 min, after the evals above):
$TG_SCORERS/venv/bin/python eval/significance.py   # -> results/SIGNIFICANCE.md + significance.json
```

Logs go to `logs/<job-name>_<jobid>.out`. Each training log prints
`replay subset: 30000 examples, md5=beea7a1bfd76`, which must be the same in every run. Trained models
go to `$TG_RUNS/<run>/final`.

### What to expect

- **The official model** uses greedy decoding, so evaluating it should reproduce the numbers in
  `results/reference/` up to tiny GPU-level differences. Its translation scores match the Tower paper's
  (COMET-22; paper in brackets):

  | FLORES en→xx | FLORES xx→en | WMT23 en→xx | WMT23 xx→en | TICO en→xx |
  |---|---|---|---|---|
  | 88.45 (88.51) | 88.27 (88.27) | 84.64 (84.28) | 81.91 (82.77) | 86.98 (87.01) |

- **Retrained models** will not match our scores exactly, because GPU training is not bit-for-bit
  deterministic. Two identical smoke runs differed by 0.40 F0.5 on Italian and 0.28 on Chinese.

## Implementation notes

- **Why continue from TowerInstruct?** Unbabel released no intermediate fine-tuning checkpoints.
  Injecting GEC halfway would mean re-running the full 4-epoch fine-tune (days of compute). Continuing
  from the released model with a replay subset is much cheaper, and the `gec_control` run separates the
  effect of GEC data from the effect of extra training.
- **Italian dev is held out.** MultiGEC-2025 publishes no references for its test split, so dev is the
  only labelled Italian evaluation set. It is never used for training.
- **The Chinese set is token-matched, not size-matched.** MuCGEC pairs were shuffled (seed 42) and taken
  until they reached the token count of the Italian training file (355,441 tokens with the TowerBase
  tokenizer). Italian has fewer examples (essays) but a similar number of tokens.
- **Tokenizer:** TowerInstruct already has the ChatML and pad tokens. `train.py` only adds tokens that
  are missing, so the embedding matrix stays the same size (32,007).
- **Scorer patches** (`patches/`, applied by `setup/3_scorers.sh`): the ChERRANT patch moves
  `from ltp import LTP` inside the word-level branch. Character-level scoring doesn't use LTP, so scores
  are unchanged. The MultiGEC patch lowercases only the output filename instead of its whole path, so
  the scorer also works when the path contains capitals. Its scores are unchanged too. `score_gec.py`
  builds the reference M2 file the MultiGEC scorer expects, using the command the upstream script has
  commented out.
- **Scorer check:** feeding the references in as outputs gives exactly 100 for Italian and F0.5 = 99.94
  for Chinese. The Chinese gap most likely comes from a newer OpenCC (traditional→simplified
  conversion) than the gold file was built with. It shifts absolute Chinese scores by about 0.1 at
  most, and equally for every run.

## Licence

The code is under the MIT licence (see [LICENSE](LICENSE)). The datasets and models keep their own
licences. MultiGEC-2025 in particular is research-only and may not be redistributed.
