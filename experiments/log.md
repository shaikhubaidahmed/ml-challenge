# Experiment log

All numbers are measured. Validation = train S1 entities against the FULL train S2/S3 pool
(same distractor density as test), unless stated otherwise.

## EXP-000 Data audit (2026-09-25)
- Train S1 2,206,821 / pool 10,320,219; test S1 1,732,544 / pool 9,969,589.
- Singletons 5.58%, one-to-one 5.40%, one-to-many 89.02%; mean 3.46 matches, max 11.
- Each S2/S3 record belongs to at most one S1 entity; 100% of positives share country.
- Test adds France (259,452 S1). Devanagari names only in India S2/S3.

## EXP-001 Blocking: single tokens (name+address), max_df 400, IDF-sum top-K
- Sample 50k train S1 vs full pool. Pair recall @60 = 0.560, F0.5 ceiling 0.668, 20.8% S1 with 0 cands.
- Diagnosis: only 70.8% of positives share a single token with df<=400; names/streets reuse a small vocabulary.
- Decision: rejected.

## EXP-002 Blocking: + compound keys (unordered name-token pairs, adjacent address bigrams), cap 500
- Key coverage of positives at cap 500: 97.8% (nn+ab alone 97.3%).
- Pair recall: @5 0.794, @10 0.882, @20 0.912, @40 0.931, @100 0.948. F0.5 ceiling @40 0.973, @100 0.980.
- S1 with zero candidates: 5 / 50,000. Runtime ~3.5 min for 50k S1 (index build dominates).
- Decision: adopted; stage-1 top_k=60. ~3% of positives share a key but are ranked out -> future rerank work.

## EXP-003 Full stage-1 blocking (cap 500, top_k 60)
- Train: 131,517,284 pairs, 2,206,542/2,206,821 S1 with candidates, 926 s.
- Test: 103,356,565 pairs, 1,731,970/1,732,544 S1 with candidates (France 258,966/259,452), 829 s.

## EXP-004 Learned Devanagari->Latin token dictionary (analysis only)
- 312,725 Devanagari-name positives; token count equals S1 token count in 99.997% -> positional alignment.
- Devanagari name vocabulary: 153 tokens; 150 with >=80% consistent Latin mapping and >=3 occurrences.
- Coverage of test Devanagari name-token occurrences (dict conf>=0.6, count>=2): 94.6%.
- Decision: adopt in normalization for v2 (requires re-blocking). Learned from train labels only.

## EXP-005 v1 model (work/, exp v1) — 2026-09-25 21:17-21:34
- Features: 34 (rapidfuzz name/address ratios, token/number Jaccard, lengths, script flag, source, blocking context).
- Train features on 30% of train S1 (662,411 entities; context from FULL blocking). Stage 1 LightGBM
  (255 leaves, lr 0.08, <=600 rounds) cross-fitted on 2 S1 folds (50% of fold entities per fit).
- Stage-1 blocking on subset: pair recall 0.9377, F0.5 ceiling 0.9756. Pruned (p1>=0.01): 4.19 cands/S1, recall 0.9365.
- OOF macro F0.5, stage 1 only: best 0.94603 @ thr 0.70 exclusive (P 0.9716, R 0.8942, singleton 0.9412).
- Stage 2 (within-S1 context only, --no_pool_ctx): best 0.94698 @ thr 0.65 exclusive (P 0.9701, R 0.8997, singleton 0.9594).
- Expected-F per-entity rule on stage-2 OOF: 0.94794 (+0.001 vs global threshold).
- Exclusive assignment helps at every threshold (+0.0003..0.003).

## EXP-006 Error analysis on v1 OOF (2,291,817 true pairs)
- 6.3% of true pairs lost before the final model (blocking/pruning), 4.0% in candidates but below threshold.
- India loses more (blocking 8.6%, model 5.1%) than US (4.9%, 3.3%). Indic-script names: blocking miss 20.6%.
- ROOT CAUSE: v1 cleaning kept only Latin + Devanagari -> Bengali/Gujarati/Tamil/Telugu/Kannada/Malayalam/
  Gurmukhi/Oriya names became EMPTY (3.2% of train pool names, 377k test pool names).
- Abbreviation table applied to names: "all"->"allee", "r"->"rue", "s"->"south" corrupt names/initials.
- Hard false positives: distractors = same address, perturbed name (Orbolayn/Orbola, JS/IS, CGU/CG, Halkara/Halcara).
- FP: pool record with empty address whose name exists on several S1 entities.
- Decision: v2 = keep all Indic blocks + learned positional dictionary for all scripts, field-specific abbreviations,
  full-train features with pool-competition stage-2 features, expected-F rule evaluated.

## EXP-007 v2 (work_v2/, exp v2) — full train, all Indic scripts, field abbreviations, pool context
- OOF over ALL 2,206,821 train S1 (cross-fitted, 2 folds): per-fold 0.96132 / 0.96096.
- Stage-1 blocking pair recall 0.9425 (v1 0.9377), F0.5 ceiling 0.9779. Pruned: 4.06 cands/S1, recall 0.9417.
- Stage 1 only: 0.95824 @0.70 (P 0.9801, R 0.9117).
- Stage 2 (with pool-competition features): 0.96114 @0.65-0.70 threshold (P 0.9816, R 0.9168, singleton 0.9799).
- Expected-F rule (extra 0.3): 0.96158 (P 0.9822, R 0.9163, singleton 0.9646) -> selected.
- Test: 7,986,245 candidates, 5,554,274 matches, 1,621,904/1,732,544 S1 matched. Validator PASS (--check-ids).
- Pipeline wall time ~6h incl. contention/pausing; training 1402 s.

## EXP-008 Blocking depth + second-pass reranker (v2 normalization, 50k/60k train S1 vs full pool)
- IDF-sum pair recall @K: 60 0.9434, 100 0.9516, 150 0.9578, 200 0.9622, 300 0.9686 (ceiling 0.9883 @300).
- Reranker = LightGBM (63 leaves, 200 rounds) on bscore, n_keys, rank, name ratio/token_set, addr ratio/token_set,
  addr_empty; trained on 30k S1 entities, evaluated on 30k held-out entities (top-300 retrieval):
  recall @20 0.9170 -> 0.9642, @50 0.9392 -> 0.9658, @60 0.9426 -> 0.9661, @100 0.9510 -> 0.9667.
- Feature cost: 14.3M pairs in 22 s.
- Decision: v3 = top-300 retrieval -> reranker keeps 50 per S1; rr + rr-rank context added as pair features.
  Note: reranker's 60k training entities are ~2.7% of train S1 (minor in-sample optimism in v3 OOF).

## EXP-009 v3 (work_v3/, exp v3) = v2 + top-300 retrieval + reranker keep-50 + rr features
- Train candidates 109,703,800 (2,206,535 S1 with cands); test 86,177,554 (1,731,924). France 258,966 S1, ~49.5/S1.
- OOF over all 2,206,821 train S1: stage 1 only 0.96646 @0.70 (P 0.9832, R 0.9301).
- Stage 2 threshold 0.65-0.70: 0.96889 (P 0.9846, R 0.9337, singleton 0.9796).
- Expected-F (extra 0.3): 0.96938 (P 0.9853, R 0.9332, singleton 0.9648) -> +0.0078 over v2.

## EXP-010 v4 (work_v3/, exp v4) = v3 candidates + name-evidence features (features_name.py)
- Added: IDF (S1+pool, per country) of shared / S1-only / pool-only name tokens (sum, max), idf_jacc,
  leftover_ratio (char similarity of unshared tokens), exact-name counts among S1 and pool (both sides).
- All statistics unsupervised, computed per split; cost ~5 s per 3M pairs.
- OOF over all 2,206,821 train S1: stage 1 only 0.97499 @0.70 (P 0.9891, R 0.9432, singleton 0.9800) (v3: 0.96646).
- Stage 2 threshold 0.70: 0.97589 (P 0.9896, R 0.9445, singleton 0.9890).
- Expected-F (extra 0.3): 0.97612 (P 0.9900, R 0.9442, singleton 0.9817) -> +0.0067 over v3.

## EXP-011 Test-like validation (train S1 19% absent -> orphan records, pool/S1 5.78 as in test)
- Hypothesis: test has 5.75 pool records per S1 vs 4.68 in train -> ~40% orphans vs 26%; v4 LB 0.9540 vs OOF 0.9761.
- v4 stage-1 models on test-like features: 0.97040 @0.70 (was 0.97499) -> explains ~0.005 of the 0.022 gap.
- v5 = trained on test-like features (1,787,288 present S1): stage 1 0.97346, final expected-F 0.97454
  (folds 0.97435 / 0.97422). +0.003 over v4 at stage 1 on the same validation.
- Remaining ~0.017 LB gap unexplained; most likely France (unseen country, no labels to measure).
- v5 test: 7,380,989 candidates, 5,584,039 matches, 1,626,754 S1 matched. Validator PASS (--check-ids).

# Submission ledger

| Day | # | Version | Val F0.5 | Key change | Public LB | Decision |
|---|---|---|---|---|---|---|
| 1 | 1 | v1 (stage2, thr 0.65, exclusive; 30% train subset) | 0.9470 (OOF, 662k S1) | first end-to-end baseline | PENDING (user upload) | file: submissions/d1_s1_v1/matching_results.tsv |
| 2 | 1 | v2 (stage2 + expected-F, full train) | 0.9616 (OOF, 2.2M S1) | Indic scripts kept + learned translit; field abbrevs; pool-competition features | PENDING (user upload) | file: submissions/d2_s1_v2/matching_results.tsv |
| 2 | 2 | v3 (reranker blocking, stage2 + expected-F) | 0.9694 (OOF, 2.2M S1) | top-300 retrieval + reranker keep-50 | PENDING (user upload) | file: submissions/d2_s2_v3/matching_results.tsv |
| 2 | 3 | v4 (v3 + name-evidence features, expected-F) | 0.9761 (OOF, 2.2M S1) | IDF name-token evidence, name ambiguity counts | 0.954021 | file: submissions/d2_s3_v4/matching_results.tsv |
| 3 | 2 | v5 (v4 trained on test-like absent-S1 data) | 0.9745 (test-like OOF) | orphan-record simulation | PENDING (user upload) | file: submissions/d3_s2_v5/matching_results.tsv |
