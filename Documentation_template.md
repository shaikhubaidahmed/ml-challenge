# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM_NAME}}
**Team Members:** {{TEAM_MEMBERS}}
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A classical, fully reproducible blocking + gradient-boosting pipeline. Records are normalized (including a
Devanagari/Bengali/Gujarati/Tamil/Telugu/Kannada/Malayalam/Gurmukhi/Oriya → Latin token dictionary learned only
from training pairs), candidates are retrieved per country with an inverted index over compound keys and
re-ranked by a small LightGBM, and a two-stage LightGBM pair classifier with name-evidence and "competition"
features scores them. Matches are decided per Source-1 entity by an expected-F0.5 rule, with each S2/S3 record
assigned to at most one Source-1 entity. Key innovations: compound blocking keys + learned reranker (blocking
recall 94.3% → 96.6% at fewer candidates), learned transliteration for nine Indic scripts, IDF name-evidence
features, and training with simulated absent Source-1 entities to match the test set's higher orphan-record rate.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the provided data (all numbers from our audit scripts):

- **Scale:** train S1 2,206,821 / S2+S3 10,320,219; test S1 1,732,544 / S2+S3 9,969,589. Naive all-pairs on test
  is 1.7×10¹³ comparisons, so blocking is mandatory.
- **Match structure:** 5.58% of train S1 entities are singletons, 5.40% one-to-one, 89.02% one-to-many
  (mean 3.46, max 11 matches). **Every S2/S3 record belongs to at most one S1 entity** (0 exceptions in 7.64M
  positive pairs) and **100% of positives share the country label**.
- **Name noise:** only 21.7% of positive pairs have an identical normalized name and 8.4% an identical normalized
  address. Observed noise: OCR-style typos (`E1ectric`, `TRIANLGE`), token shuffles/brackets
  (`(Clinic) College Brand`), legal-suffix changes, domain-style names (`brandcollegeclinic.com`), junk prefixes
  (`--`, `<<`), `<NULL>` inside addresses, dropped/changed house numbers, reordered address components.
- **Scripts:** India S2/S3 names appear in nine Indic scripts (Devanagari 4.1% of all pool names; Bengali,
  Gujarati, Tamil, Telugu, Kannada, Malayalam, Gurmukhi, Oriya together another 3.2%). S1 is always Latin.
  An Indic-script name has the same number of tokens as its S1 name in 99.997% of positives.
- **Ambiguity:** 30% of S1 names are shared by more than one S1 entity in the same country; 3.3% of S2/S3
  addresses are empty.
- **Hard negatives:** unmatched S2/S3 records (26% of the train pool) are mostly decoys: the same address with one
  altered name token (`Orbola Bright Holdings` vs `Orbolayn Bright`, `IS Real LLC` vs `JS LLC Real`).
- **Train/test shift:** test has 5.75 S2/S3 records per S1 (every country) vs 4.68 in train. With the same
  matches per entity, ~40% of test pool records have no S1 owner vs 26% in train, i.e. test contains many
  *orphan* records whose entity is not in Source 1. France (15% of test S1) is absent from training.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned reranker + two-stage gradient-boosted pair classifier + per-entity decision rule.
**Core Innovation:** compound-key blocking with a learned reranker; training-data-only transliteration for nine
Indic scripts; IDF name-evidence features; exclusive assignment + expected-F0.5 decisions; absent-S1 simulation.

Pipeline phases: (A) normalization + candidate generation → (B) pair features → (C) stage-1 pair model →
learned pruning → stage-2 context model → (D) decision → (E/F) submission files → (G) official validator.

**Normalization** (deterministic, several representations kept): lowercase; Unicode NFKD with Latin combining
marks removed (Indic vowel signs preserved); `&`→`and`; `<NULL>`/`null`/`n/a` removed; punctuation → space;
Indic tokens mapped through the learned dictionary (names: positional alignment with the S1 name; addresses:
co-occurrence lift), unseen Devanagari tokens romanized by generic character rules; business-name abbreviations
(`pvt`→`private`, `ltd`→`limited`, …) applied only to names and address abbreviations (`rd`→`road`, `r`→`rue`,
`nr`→`near`, …) only to addresses (applying them to both corrupted names such as `All`→`allee`).

**Country** is used only as an open-set partition key for blocking and statistics (never one-hot), so France is
handled exactly like the training countries.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** within each country, every record yields hashed keys: single name tokens, single
  address tokens, **unordered pairs of name tokens**, and **adjacent address-token bigrams**. Keys occurring in
  more than 500 pool records (per country) are dropped. Single-token blocking alone only reached 56% pair
  recall@60 (the name/street vocabulary is small); compound keys raised key coverage of true pairs to 97.8%.
- **First pass:** candidate score = sum of IDF of shared keys; keep the top 300 per S1 entity.
- **Second pass (learned reranker):** a LightGBM (63 leaves) on 8 cheap features (IDF score, #shared keys,
  rank, name ratio / token-set ratio, address ratio / token-set ratio, empty-address flag), trained on 60k train
  S1 entities, keeps the top 50 per S1. Held-out recall: @20 0.9170 → 0.9642, @50 0.9392 → 0.9658, @60 0.9426 → 0.9661.
- **Learned pruning:** the stage-1 pair model drops pairs with probability < 0.01. The remaining pairs are exactly
  what the final (stage-2) model scores and are written to `candidate_pairs.tsv`.
- **Candidate pairs generated (test):** 86,177,554 after the reranker (1,731,924 of 1,732,544 S1 with candidates);
  7,380,989 in the final `candidate_pairs.tsv`.
- **How true matches were not lost:** measured candidate recall at every stage on train S1 against the full train
  pool (same distractor density as test). Final candidate-set pair recall 0.9648 (v5, test-like train validation), F0.5 ceiling
  0.987.

---

## 4. Matching Model

**Features used (stage 1, 52 features):**
- Name: rapidfuzz ratio, token-sort, token-set, partial, Jaro-Winkler; ratio/partial on space-free names (domain
  names); token Jaccard and shared-token count; first-token equality; lengths; Indic-script flag.
- Name evidence (IDF over S1+pool names per country): IDF sum/max of shared, S1-only and pool-only tokens,
  IDF-weighted Jaccard, character similarity of the unshared tokens (typo vs different word), number of S1 and
  pool records carrying exactly the same name (ambiguity).
- Address: ratio, token-sort, token-set, partial; token Jaccard; house-number Jaccard, count, subset flag; lengths.
- Blocking context: IDF score, #shared keys, reranker score; rank and gap of the pair among the S1 entity's
  candidates and among the S1 entities competing for the same pool record; #S1 competing for the record; source.

**Stage 2 features (22):** 12 key stage-1 features (IDF score and ranks, name/address ratios, Jaccards, script flag, source, address length) plus the stage-1 probability and its context: rank/gap within the S1 entity, rank/gap within the
pool record, sum of probabilities per S1 and per record, #candidates above 0.5, best competing S1 probability.

**Model type:** LightGBM binary classifiers (stage 1: 255 leaves, lr 0.08, ≤600 rounds, early stopping;
stage 2: 127 leaves, lr 0.05). **2-fold cross-fitting by Source-1 entity**, so every train entity is scored by
a model that never saw it; test predictions average both fold models. All models are MIT-licensed LightGBM,
far below 8B parameters. No pretrained language models, no external data.

**Threshold selection method:** macro F0.5 on out-of-fold predictions. Each pool record is first assigned to its
highest-scoring S1 entity only (exclusive assignment; helps at every threshold). Then, per S1 entity, the top-k
candidates that maximize approximate expected F0.5, E[F](k) ≈ 1.25·Σ_{i≤k} p_i / (k + 0.25·(Σ p + 0.3)), are
chosen, compared with the probability that the entity is a singleton, Π(1−p_i). This beat the best global
threshold on every version and was selected automatically.

---

## 5. Results & Error Analysis

Out-of-fold macro F0.5 over **all 2,206,821 train S1 entities** (both folds agree to ±0.0002):

| Version | Change | OOF F0.5 | Precision | Recall | Singleton acc. | Public LB |
|---|---|---|---|---|---|---|
| v1 | baseline (30% subset, Latin+Devanagari only) | 0.9470 | 0.970 | 0.900 | 0.959 | {{LB_V1}} |
| v2 | all Indic scripts + learned transliteration, field-specific abbreviations, full train | 0.9616 | 0.982 | 0.916 | 0.965 | {{LB_V2}} |
| v3 | top-300 retrieval + learned reranker | 0.9694 | 0.985 | 0.933 | 0.965 | {{LB_V3}} |
| v4 | + name-evidence features | 0.9761 | 0.990 | 0.944 | 0.982 | 0.9540 |
| v5 | + training with 19% of S1 absent (test-like orphans) | 0.9745 (test-like; v4 on same: 0.9704 stage 1) | 0.989 | 0.943 | 0.978 | {{LB_V5}} |

- **F0.5 Score (macro):** best validation {{BEST_OOF}}; public leaderboard {{BEST_LB}}.
- **Common false positives (wrong merges):** 87% of v2 false merges were decoy records (same address, one altered
  name token); pool records with an empty address whose name is shared by several S1 entities were 4× over-
  represented. On test, orphan records of entities absent from Source 1 cause additional false merges; this is the
  main cause of the gap between validation (0.976) and the public leaderboard (0.954) for v4.
- **Common false negatives (missed matches):** v2: 5.8% of true pairs never reached the model (ranked below the
  IDF top-60; fixed largely by the reranker), 2.9% were scored below the cut, 32% of those with an empty
  address (name is the only evidence). Indic-script records were the worst group before transliteration.

---

## 6. Conclusion

Careful blocking (compound keys + a learned reranker) and training-data-only normalization (transliteration of
nine Indic scripts) mattered more than model complexity: a two-stage LightGBM with name-evidence and competition
features reached 0.976 out-of-fold macro F0.5. The main lesson is that the validation split must reproduce the
test set's structure: test contains far more orphan S2/S3 records than train, which cost ~0.02 F0.5 until modeled.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md`. Entry point: `src/scripts/run_final.sh`.

| Path (under `src/`) | Purpose |
|---|---|
| `ber/data.py` | TSV loading (tab, quoting disabled), parquet caches, ground truth → labels |
| `ber/normalize.py`, `ber/translit.py` | normalization; transliteration dictionary learned from train pairs |
| `ber/blocking.py`, `ber/rerank.py` | per-country compound-key inverted index; learned reranker |
| `ber/features.py`, `ber/features_name.py` | pair similarity, context, and name-evidence features |
| `ber/model.py`, `ber/decision.py` | LightGBM stages, stage-2 context, exclusive assignment, expected-F rule |
| `ber/metrics.py`, `ber/submission.py` | official macro F0.5, blocking recall ceiling; TSV writers |
| `scripts/*.py`, `scripts/run_final.sh` | pipeline steps and the end-to-end driver |
| `configs/*.yaml` | paths, blocking and validation parameters per version |
| `experiments/log.md` | full experiment log and submission ledger |

### B. Additional Results

Blocking recall of true pairs (50k train S1 vs full train pool): IDF top-K @60 0.9434, @100 0.9516, @300 0.9686;
with reranker keep-50: 0.9658. v2 transliteration dictionary: 161 learned tokens (Devanagari), extended to all
Indic scripts in v2; covers 97.4% of Indic-script token occurrences in test. Runtime on a 10-core Apple M5 (16 GB):
blocking ≈ 30 min (train) / 26 min (test), features ≈ 9 + 4 min per split, training ≈ 25 min, test inference ≈ 40 min.

---

**Note:** All reported numbers were measured by the scripts in this package; no external data or services were used.
