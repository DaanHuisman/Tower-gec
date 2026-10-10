# Adding GEC to Tower's training mix

**Research question.** If grammatical error correction (GEC) data is added to the training mix of a
translation LLM, does GEC performance go up, and what does it cost? Concretely: can
[TowerInstruct-7B](https://huggingface.co/Unbabel/TowerInstruct-7B-v0.1) learn GEC for **Italian**
(little GEC data) and **Chinese** (a different writing system) at the same time, and what happens to
its translation quality?

The Tower authors themselves suggest that TowerInstruct's average GEC performance comes from the
absence of GEC data in TowerBlocks. This repo tests that.

## Design

All runs continue training the released TowerInstruct-7B for a short extra phase. They use the same
**fixed random subset of 30k TowerBlocks examples** (the "replay" data, so the model keeps practising
its existing tasks). The only difference between runs is the GEC data added on top:

| Run | Training data | Question it answers |
|---|---|---|
| `gec_control` | replay only | What does extra training by itself do? |
| `gec_it` | replay + Italian GEC (651 essays) | GEC gain for a language with little GEC data |
| `gec_zh` | replay + Chinese GEC (2,610 sentence pairs) | GEC gain for a different writing system |
| `gec_it_zh` | replay + both | Can one model learn both? Do they interfere? |
| official TowerInstruct-7B | no extra training | Reference point |

`gec_it_zh` was evaluated on GEC (own dev sets) and translation only (`results/gec_it_zh.*`). It was not
run on Tower's other tasks or the paper's GEC sets, and it is not in the significance tests.

Every model is evaluated on:
- **GEC:** Italian MultiGEC-2025 dev (81 essays; GLEU and ERRANT P/R/F0.5) and Chinese MuCGEC dev
  (1,137 sentences; ChERRANT P/R/F0.5). Both use the official scorers.
- **Translation:** FLORES-200, WMT23 and TICO-19 (COMET-22 and chrF). This is where a cost would show up.
- **Tower's other tasks** (separate suite, `./submit.sh tasks`): automatic post-editing of NLLB-3.3B
  translations of WMT23 (6 language pairs; COMET-22, chrF, TER) and named-entity recognition on
  MultiCoNER II (7 languages; entity F1). Data and prompts are TowerEval-Data-v0.1, the set the Tower
  paper uses, and scoring follows tower-eval.
- **The Tower paper's own GEC sets** (separate suite, `./submit.sh gec_paper`): English CoNLL-2014,
  German Falko-MERLIN and Spanish COWSL2H with the paper's 5-shot prompts, scored with edit rate
  (TER against the reference) and official ERRANT. These show whether Italian/Chinese GEC training
  carries over to GEC in other languages.

Training settings: 2 epochs, learning rate 2e-6 with cosine decay and 20 warmup steps, batch size 256,
full fine-tune on 4 H100s (about 1–1.5 h per run). The loss covers only the assistant turns.

## Setup (one time, on a Snellius login node)

### 0. Request MultiGEC access first (it takes time)

The Italian data comes from the MultiGEC-2025 release. Its terms forbid redistribution, so **it is not
in this repo, and each of us has to request access individually** at
<https://lt3.ugent.be/resources/multigec-dataset>. The terms are research use only, no sharing, and
no feeding the data to proprietary models that retain it for training.

Once approved, unpack the release so that this file exists:
`$TG_DATA/multigec-2025-participants-1/italian/Merlin/it-merlin-orig-train.md`
(with the default config, `$TG_DATA` is `~/data/TowerBlocks`).

### 1–5. Install, download, build, verify

```bash
git clone <this repo> tower-gec && cd tower-gec
bash setup/1_env.sh          # Python 3.9 venv with pinned packages               (~10 min)
bash setup/2_download.sh     # TowerInstruct-7B, TowerBlocks, MuCGEC, COMET-22     (~15 min, 28 GB)
bash setup/3_scorers.sh      # ChERRANT + MultiGEC scorers in a Python 3.12 venv    (~10 min)
bash setup/3b_errant.sh      # official ERRANT for the paper's GEC sets, own venv   (~1 min)
bash setup/4_build_data.sh   # GEC training sets + evaluation sets                  (~2 min)
source config.sh && $TG_ENV/bin/python setup/verify.py
```

`verify.py` must end with **"All checks passed."** It checks the following against the reference setup:
- every downloaded file (md5)
- the content of the built GEC sets
- the evaluation sets
- the fingerprint of the 30k replay subset (`beea7a1bfd76`)

If all of these match, your runs use exactly the same data as everyone else's.

All scripts skip anything that already exists, so they are safe to rerun. They never overwrite data.

**Where things go.** Default locations are set in [config.sh](config.sh). To use other locations,
export the variable before running anything, e.g. `export TG_MODELS=/scratch-shared/$USER/models`.

| Variable | Default | Contents |
|---|---|---|
| `TG_DATA` | `~/data/TowerBlocks` | TowerBlocks, MuCGEC, MultiGEC, built GEC sets (1.2 GB) |
| `TG_MODELS` | `~/models` | TowerInstruct-7B (26 GB), TowerBase-7B tokenizer |
| `TG_ENV` | `~/tower-env` | training/eval venv |
| `TG_SCORERS` | `~/gec-scorers` | GEC scorers + their venvs |
| `TG_RUNS` | `/scratch-shared/$USER/runs` | trained models, about 51 GB each |
| `HF_HOME` | `/scratch-shared/$USER/hf_cache` | Hugging Face cache |

Snellius purges old files from scratch. Copy trained models you want to keep.

## Running

Start with the smoke test. It checks the full pipeline in about 20 minutes: a 3-minute training run,
then generation and scoring on the GEC sets.

```bash
./submit.sh train smoke                                     # prints "... job <id> ..."
./submit.sh eval gec_smoke gec --dependency=afterok:<id>    # starts when training succeeds
cat results/gec_smoke.gec_summary.txt                       # when both jobs are done (squeue -u $USER)
```

The smoke model has barely changed, so its GEC scores should be within about ±0.5 of the official
model's (see below). Training on GPUs is not bit-for-bit deterministic, so reruns of the same training
job differ slightly. For example, two smoke runs gave Italian F0.5 46.16 and 46.56. Evaluating a fixed
model (`./submit.sh eval official`) uses greedy decoding, so it should match the reference numbers up
to tiny GPU-level differences. The replay fingerprint and the data checks must match exactly.

The real experiments:

```bash
./submit.sh all                       # the 4 training runs, each followed by its eval
./submit.sh eval official             # the official TowerInstruct-7B (reference numbers below)

# or one at a time:
./submit.sh train it                  # control | it | zh | it_zh
./submit.sh eval gec_it               # mt | gec | mt,gec (default)

# Tower's other tasks (APE + NER) for official, gec_control, gec_it and gec_zh (~1.5 h each):
./submit.sh tasks
./submit.sh gec_paper                      # the paper's GEC sets (en/de/es) for the same four (~30 min each)
$TG_ENV/bin/python eval/compare_tasks.py   # side-by-side tables once the jobs are done

# paired bootstrap significance tests for every comparison (CPU, ~1 min, after the evals above):
$TG_SCORERS/venv/bin/python eval/significance.py   # -> results/SIGNIFICANCE.md + significance.json

# hyperparameter overrides for exploratory runs (use a different name so results don't mix):
REPLAY=20000 EPOCHS=1 ./submit.sh train it
```

- **Logs:** `logs/<job-name>_<jobid>.out`. Each training log prints
  `replay subset: 30000 examples, md5=beea7a1bfd76`, which must be the same in every run.
- **Results in `results/`:**
  - `<run>.summary.txt`: translation, COMET-22 and chrF per language pair and direction
  - `<run>.gec_summary.txt` / `<run>.gec_scores.json`: GEC
  - `<run>.tasks_summary.txt` / `<run>.tasks_scores.json`: APE and NER
  - `<run>.gec_paper_summary.txt` / `<run>.gec_paper_scores.json`: the Tower paper's GEC sets
  - `SIGNIFICANCE.md` / `significance.json`: paired bootstrap tests between the four models
  - `PAPER_TESTS.md`: every evaluation in the Tower paper, whether we ran it, and our numbers next to the paper's
- **Trained models:** `$TG_RUNS/<run>/final`

Only the summary and score files belong in git; `.gitignore` already excludes the model outputs,
which contain MultiGEC text.

## Reference numbers (official TowerInstruct-7B)

Full files are in [results/reference/](results/reference). Your `./submit.sh eval official` should
reproduce them.

GEC:

| Test set | GLEU | P | R | F0.5 | Outputs left unchanged |
|---|---|---|---|---|---|
| Chinese (MuCGEC dev) | – | 27.24 | 14.57 | 23.20 | 34.9% |
| Italian (MultiGEC dev) | 58.82 | 54.95 | 27.47 | 45.79 | 3.7% |

The model mostly under-corrects. It leaves 35% of Chinese sentences unchanged, while only about 5%
of them are error-free.

Translation, COMET-22 (the paper's numbers are in brackets):

| | FLORES en→xx | FLORES xx→en | WMT23 en→xx | WMT23 xx→en | TICO en→xx |
|---|---|---|---|---|---|
| TowerInstruct-7B | 88.45 (88.51) | 88.27 (88.27) | 84.64 (84.28) | 81.91 (82.77) | 86.98 (87.01) |

## Repository layout

```
config.sh                  paths (the only place to change locations)
submit.sh                  submit training / evaluation jobs (experiments are defined here)
train.py                   full fine-tune; --replay_samples, --extra_data folder[:N] (N = upsample)
slurm/                     train_gec.sbatch, eval.sbatch, train_baseline.sbatch (optional, days)
data/build_gec_it.py       MultiGEC Italian -> TowerBlocks format (train 651 / dev 81, dev held out)
data/build_gec_zh.py       MuCGEC -> token-matched Chinese training sample (2,610 pairs)
eval/prepare_gec_data.py   GEC eval set (Italian dev + MuCGEC dev)
eval/prepare_data.py       MT eval set (FLORES-200, WMT23, TICO-19)
eval/prepare_tasks_data.py APE + NER and the paper's GEC eval sets (TowerEval-Data-v0.1)
eval/generate.py           greedy decoding with the Tower chat template
eval/score.py              COMET-22 + chrF
eval/score_gec.py          ChERRANT (zh) + MultiGEC GLEU/ERRANT (it)
eval/score_tasks.py        APE (COMET-22, chrF, TER) + NER (entity F1, as tower-eval)
eval/score_gec_paper.py    the paper's GEC sets: edit rate (TER) + official ERRANT, as tower-eval
eval/compare_tasks.py      APE, NER and paper-GEC scores of several runs side by side
eval/significance.py       paired bootstrap tests (control vs official, it/zh vs control, ...) on all metrics
eval/m2_counts.py          per-sentence ERRANT/ChERRANT counts, for the bootstrap
setup/                     one-time setup steps + verify.py
requirements/              exact package versions for both venvs
patches/                   one-line patches for ChERRANT and the MultiGEC scorer (see below)
results/                   score summaries of every run, SIGNIFICANCE.md, PAPER_TESTS.md
results/reference/         official TowerInstruct-7B scores
references.bib             the paper's bibliography (models, data, scorers, related work)
```

## Details worth knowing

- **Why continue from TowerInstruct?** Unbabel released no intermediate fine-tuning checkpoints.
  Injecting GEC halfway would mean re-running the full 4-epoch fine-tune ourselves (days of compute).
  Continuing from the released model with a replay subset is much cheaper, and the `gec_control`
  run separates the effect of GEC data from the effect of extra training.
- **Italian dev is held out.** MultiGEC-2025 publishes no references for its test split, so dev is the
  only labelled Italian evaluation set. It is never used for training.
- **The Chinese set is token-matched, not size-matched.** MuCGEC pairs were shuffled (seed 42) and taken
  until they reached the token count of the original Italian file (355,441 tokens with the TowerBase
  tokenizer). Italian has fewer examples (essays) but a similar number of tokens.
- **Upsampling:** `train.py --extra_data $TG_DATA/gec_it:3` repeats each Italian example 3 times.
  The main experiments do not use it; the smoke test does, to exercise the code path.
- **Tokenizer:** TowerInstruct already has the ChatML and pad tokens. `train.py` only adds tokens that
  are missing, so the embedding matrix stays the same size (32,007).
- **Scorers:** the ChERRANT patch moves `from ltp import LTP` inside the word-level branch.
  Character-level scores don't use LTP, so they are unchanged. The MultiGEC patch lowercases only the
  output filename instead of its whole path, so the scorer also works when the repo path contains
  capitals. Its scores are unchanged too. `score_gec.py` builds the reference M2 file it expects, using the command the upstream script has
  commented out.
- **Scorer check:** feeding the references in as outputs gives exactly 100 for Italian and F0.5 = 99.94
  for Chinese. The Chinese gap most likely comes from a newer OpenCC (traditional→simplified
  conversion) than the gold file was built with. It shifts absolute Chinese scores by about 0.1 at
  most, and does so equally for every run.

## Data licences

| Data | Licence | In this repo? |
|---|---|---|
| TowerBlocks | per-source licences (see its dataset card) | no, downloaded |
| MuCGEC | see github.com/HillZhang1999/MuCGEC | no, downloaded |
| MultiGEC-2025 (MERLIN Italian) | research only, **no redistribution**, request access individually | **never commit it or anything built from it** |

The code in this repo is under the MIT licence (see [LICENSE](LICENSE)). That licence does not cover
the data or models above.
