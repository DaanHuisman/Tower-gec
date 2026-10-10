# Tower paper tests vs. our results

Checklist of every evaluation in the Tower paper (Alves et al. 2024, arXiv:2402.17733) and whether we
ran it. Our numbers come from `results/<run>.summary.txt` (translation), `results/<run>.tasks_summary.txt`
(APE + NER) and `results/<run>.gec_paper_summary.txt` (the paper's GEC sets), produced on 2026-10-09.
Significance tests are in `results/SIGNIFICANCE.md`. "Paper" is the paper's TowerInstruct-7B.

- **official**: TowerInstruct-7B-v0.1 as released
- **control**: + 2 epochs on the 30k replay subset
- **it**: + replay + Italian GEC
- **zh**: + replay + Chinese GEC

Our own Italian and Chinese GEC tests are not in the paper; see `RESULTS.txt`.

## Checklist

| # | Paper test | Paper section | Status |
|---|---|---|---|
| 1 | Translation, FLORES-200, COMET-22 | 3.2, Table 1 | ✅ Done |
| 2 | Translation, WMT23, COMET-22 | 3.2, Table 1 | ✅ Done |
| 3 | Translation, TICO-19, COMET-22 | 3.2, Table 1 | ✅ Done |
| 4 | Translation, chrF | App. F | ✅ Done (per language pair, in `<run>.summary.txt`) |
| 5 | Translation, xCOMET, CometKiwi-22, BLEURT | App. F | ❌ **Missing** |
| 6 | APE: post-editing NLLB-3.3B on WMT23, COMET-22 | 3.3, Tables 3, 21, 23 | ✅ Done (we also have chrF and TER) |
| 7 | APE: how often the model edits | 3.3, Fig. 6, App. B | ⚠️ Partly: we have edit rates, but not the comparison with GPT-4 |
| 8 | NER: MultiCoNER 2023, sequence F1, 7 languages | 3.3, Tables 3, 25 | ✅ Done |
| 9 | GEC: CoNLL-2014 (en), COWSL2H (es), mlconvgec2018 (de), 5-shot, edit rate | 3.3, Tables 3, 24 | ✅ Done |
| 10 | GEC: ERRANT on the same sets | App. G, Table 22 | ✅ Done |
| 11 | Significance tests | 3.1 | ✅ Done, as pairwise paired bootstrap tests between our four models, for every task (`SIGNIFICANCE.md`) |
| 12 | Win rate by source length vs GPT-4 / ALMA-R | 3.2, Fig. 4 | ➖ Not applicable (compares against other systems) |
| 13 | Other models as baselines (GPT-4, Mixtral, LLaMA-2 70B, NLLB-54B, ALMA-R, …) | 3.1, App. F | ➖ Not applicable (we compare our four models) |
| 14 | Training-recipe ablations | 4, Table 4, Figs. 7–8 | ➖ Not applicable (needs retraining TowerBase / TowerInstruct) |
| 15 | Beam search and MBR decoding | App. A, Table 5 | ➖ Not done (we use greedy decoding, like the paper's main results) |

**Summary.** We have every test of the paper's that applies to our question. Only the **extra
translation metrics** (row 5) are missing, plus the GPT-4 part of row 7.

The paper's significance procedure goes further than pairwise tests: it groups many systems into quality
clusters. With only four models of our own, pairwise tests answer the same question: which differences
are real.

The bootstrap only covers the choice of test sentences, not the run-to-run variation of training (one
run per condition; two identical training runs differed by 0.40 Italian F0.5). Significant differences
much smaller than that, such as the translation and APE ones below, could still be training noise.

## Main takeaways

- **The official model reproduces the paper** to within about 1 point almost everywhere.
  - The unedited NLLB translations and the uncorrected GEC sources score exactly what the paper reports,
    so test data and scoring are the same.
  - Small differences are expected because the paper generated with vLLM and we use Hugging Face
    transformers.
  - Largest gaps: GEC edit rate in Spanish (15.39 vs 17.23, better than the paper), WMT23 xx→en (−0.86
    COMET), APE de-en (−1.39), APE en-de (−1.00) and NER Italian (−1.09 F1).
- **Italian GEC data also improves GEC in English and German.** On the paper's GEC sets, ERRANT F0.5 rises
  by +3.77 (English) and +2.71 (German) over the control, both significant (p ≤ .002). Spanish doesn't
  change (−0.21, not significant). Chinese GEC data changes nothing significantly on these sets.
- **Every GEC result in `RESULTS.txt` that looked real is significant:**
  - Italian data on Italian: F0.5 +12.57 and GLEU +16.22 over the control.
  - Chinese data on Chinese: F0.5 +4.45, with precision up and recall down.
  - Italian data on Chinese: F0.5 +2.57.
  - Chinese data on Italian (+0.87) is not significant.
- **GEC data has almost no cost elsewhere, with one small exception.** Compared with the control:
  - Translation: no significant drop for either GEC run. The Chinese run is even slightly better (+0.04
    to +0.13 COMET, significant on 4 of 5 test sets).
  - NER: no significant difference for either run.
  - APE: no difference for Italian data (+0.01). Chinese data costs **0.20 COMET** on average (p ≤ .002),
    mostly on en-zh (−0.56).
- **The extra training itself has small but significant effects** (control vs official):
  - Translation drops by 0.07–0.22 COMET on 4 of 5 test sets.
  - APE improves by 0.63 COMET.
  - NER drops by 0.64 F1.
  - Italian GEC F0.5 drops by 2.00 (p = .028).

## 1. Translation (COMET-22)

| Model | FLORES en→xx | FLORES xx→en | WMT23 en→xx | WMT23 xx→en | TICO en→xx |
|---|---|---|---|---|---|
| Paper | 88.51 | 88.27 | 84.28 | 82.77 | 87.01 |
| official | 88.45 | 88.27 | 84.64 | 81.91 | 86.98 |
| control | 88.34 | 88.21 | 84.42 | 81.85 | 86.88 |
| it | 88.36 | 88.21 | 84.43 | 81.86 | 86.90 |
| zh | 88.39 | 88.22 | 84.54 | 81.92 | 86.91 |

**Missing:** xCOMET, CometKiwi-22 and BLEURT. chrF per language pair is in `<run>.summary.txt`.

## 2. Automatic post-editing (COMET-22)

Each model post-edits NLLB-3.3B translations of the WMT23 test set. Post-editing helps if the result
beats the unedited NLLB row.

| Language pair | Paper | NLLB (no edits) | official | control | it | zh |
|---|---|---|---|---|---|---|
| en-de | 81.86 | 77.87 | 80.86 | 81.61 | 81.66 | 81.54 |
| en-ru | 83.92 | 82.93 | 83.86 | 84.00 | 83.99 | 83.88 |
| en-zh | 83.52 | 75.72 | 83.41 | 84.12 | 84.24 | 83.55 |
| de-en | 82.29 | 79.92 | 80.90 | 82.59 | 82.56 | 82.45 |
| ru-en | 80.82 | 80.05 | 80.86 | 80.94 | 80.92 | 80.85 |
| zh-en | 77.45 | 76.44 | 77.32 | 77.72 | 77.66 | 77.48 |

The NLLB column matches the paper's Table 23 exactly.

The paper reports APE averaged per direction, in two ways:

| Average | Paper | NLLB | official | control | it | zh |
|---|---|---|---|---|---|---|
| Table 21, en→xx (en-de, en-ru, en-zh) | 83.10 | 78.84 | 82.71 | 83.24 | 83.29 | 82.99 |
| Table 21, xx→en (de-en, ru-en, zh-en) | 80.19 | 78.81 | 79.70 | 80.42 | 80.38 | 80.26 |
| Table 3, en→xx (en-de, en-zh only) | 82.69 | 76.79 | 82.14 | 82.86 | 82.95 | 82.55 |
| Table 3, xx→en (de-en, ru-en only) | 81.56 | 79.99 | 80.88 | 81.77 | 81.74 | 81.65 |

The paper only reports COMET-22 for APE. We also measured chrF and TER (average over 6 pairs), and the
share of segments each model changed:

| | NLLB | official | control | it | zh |
|---|---|---|---|---|---|
| chrF ↑ | 49.10 | 51.00 | 52.66 | 52.58 | 52.34 |
| TER ↓ | 62.27 | 60.91 | 59.28 | 59.80 | 60.09 |
| Segments edited | – | 35.5% | 41.2% | 40.9% | 39.0% |

Our en-zh TER splits Chinese into characters, unlike tower-eval's settings, so it can't be compared
with tower-eval's en-zh TER.

The paper says TowerInstruct-13B edits about 30% of WMT23 zh→en segments and GPT-4 about 90% (Fig. 6).
Our 7B models edit 14–22% of zh-en segments. We have no GPT-4 comparison.

## 3. Named-entity recognition (F1)

MultiCoNER 2023 test, 1,000 sentences per language, entity-level F1 over 6 entity types.

| Language | Paper | official | control | it | zh |
|---|---|---|---|---|---|
| en | 75.09 | 74.38 | 73.27 | 73.46 | 72.88 |
| de | 78.01 | 78.05 | 77.30 | 77.84 | 77.29 |
| es | 74.89 | 74.32 | 74.01 | 74.00 | 73.71 |
| fr | 70.35 | 70.80 | 70.23 | 70.41 | 70.27 |
| it | 76.39 | 75.30 | 74.85 | 74.94 | 74.94 |
| pt | 73.88 | 73.51 | 72.00 | 72.19 | 72.46 |
| zh | 53.13 | 53.02 | 53.25 | 53.40 | 52.49 |
| **average** | **71.68** | **71.34** | **70.70** | **70.89** | **70.58** |

Badly formatted answers (unreadable, or the wrong number of tokens), out of 7,000: official 83,
control 107, it 102, zh 97. Chinese has the most (31–45 per model).

The paper's text also lists Russian for NER, but its Table 25 and the released test data only have
these 7 languages.

## 4. GEC in English, German and Spanish

The paper treats GEC as a held-out task: TowerBlocks has no GEC data, so it uses 5 examples in the
prompt. Test sets: CoNLL-2014 (English, 1,312 sentences), Falko-MERLIN from mlconvgec2018 (German,
2,337) and COWSL2H (Spanish, 821). Scored as in tower-eval.

**Edit rate** (TER against the reference, lower is better):

| | en | de | es | average |
|---|---|---|---|---|
| No edits (ours = paper) | 13.75 | 18.23 | 18.00 | 16.66 |
| Paper | 13.39 | 14.77 | 17.23 | 15.13 |
| official | 13.33 | 14.60 | 15.39 | 14.44 |
| control | 13.31 | 14.70 | 15.52 | 14.51 |
| it | 13.24 | 14.36 | 15.53 | 14.38 |
| zh | 13.23 | 14.85 | 15.60 | 14.56 |

**ERRANT F0.5** (higher is better; the paper only gives the average, 0.42 = 42):

| | en | de | es | average |
|---|---|---|---|---|
| Paper | | | | 42.00 |
| official | 41.82 | 52.26 | 41.56 | 45.21 |
| control | 42.47 | 52.02 | 41.09 | 45.19 |
| it | **46.24** | **54.73** | 40.88 | **47.28** |
| zh | 42.31 | 51.39 | 40.81 | 44.84 |

Significance against the control:
- **Italian GEC data, ERRANT F0.5:** improves English (+3.77) and German (+2.71) GEC (p ≤ .002 for
  both), but not Spanish (−0.21, p = .79).
- **Italian GEC data, edit rate:** the German change (−0.34) is significant. The average (−0.13) just
  misses (p = .054).
- **Chinese GEC data:** no significant change on any of these sets.

Edit rate moves much less than ERRANT, because most sentences need few or no edits: copying the source
already scores 13.75 in English.

ERRANT here uses the official English-only model for German and Spanish too, as tower-eval does, so
treat the German and Spanish ERRANT numbers as approximate. Feeding in the references as outputs gives
F0.5 76–89, not 100, because the references are re-tokenised before alignment.

## 5. Not done, and not needed for our question

- **Extra translation metrics** (xCOMET, CometKiwi-22, BLEURT): the paper says the trends largely match
  COMET-22, so they are optional.
- **Other baseline models, recipe ablations, decoding methods, and win rates by length:** these are
  about how Tower was built or how it compares with other systems, not about adding GEC data.
