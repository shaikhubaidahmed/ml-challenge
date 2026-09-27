#!/usr/bin/env bash
# End-to-end reproduction of the final submission (data -> blocking -> matching -> output -> validation).
#
# Usage (from the business_entity_resolution/ folder, or src/ in the submission package layout):
#   scripts/run_final.sh [config] [exp] [absent_frac]
#     config       default configs/v5.yaml (set paths.data_dir to the folder holding train/ and test/)
#     exp          default v5
#     absent_frac  default 0.81 = fraction of train S1 kept PRESENT; the rest are treated as absent so their
#                  S2/S3 records become orphans (test-like pool/S1 ratio). Use 1.0 to reproduce v4.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/v5.yaml}
EXP=${2:-v5}
KEEP=${3:-0.81}
PY=${PY:-python}

# Phase A: normalization caches + learned transliteration (train only), blocking reranker, candidates
$PY scripts/train_reranker.py --config "$CFG"                 # builds train caches + translit.json, trains reranker
$PY scripts/run_blocking.py --config "$CFG" --split train     # top-300 retrieval -> reranker keeps 50 per S1
$PY scripts/run_blocking.py --config "$CFG" --split test

# Phase B: pair features (+ name-evidence features)
if [ "$KEEP" = "1.0" ]; then
  $PY scripts/build_features.py --config "$CFG" --split train --out train_feats
  $PY scripts/add_name_features.py --config "$CFG" --split train --src train_feats --dst train_feats_final
else
  $PY scripts/build_features.py --config "$CFG" --split train --s1_frac "$KEEP" --absent_s1 --out train_feats
  $PY scripts/add_name_features.py --config "$CFG" --split train --src train_feats --dst train_feats_final \
      --present_s1_frac "$KEEP"
fi
$PY scripts/build_features.py --config "$CFG" --split test --out test_feats
$PY scripts/add_name_features.py --config "$CFG" --split test --src test_feats --dst test_feats_final

# Phases C-D: cross-fitted stage-1/stage-2 LightGBM, OOF macro F0.5, decision rule selection
$PY scripts/train.py --config "$CFG" --exp "$EXP" --feats train_feats_final --s1_frac "$KEEP" --sample 0.25

# Phases E-F: test inference -> output_dir/matching_results.tsv + candidate_pairs.tsv
$PY scripts/predict.py --config "$CFG" --exp "$EXP" --feats test_feats_final

# Phase G: official validator
OUT=$($PY -c "import sys; sys.path[:0]=['src','.']; from ber.config import load_config; print(load_config('$CFG')['paths']['output_dir'])")
TESTDIR=$($PY -c "import sys; sys.path[:0]=['src','.']; from ber.config import load_config; print(load_config('$CFG')['paths']['data_dir'])")/test
VALIDATOR=${VALIDATOR:-../../student_resource/utils/validate_submission.py}
if [ -f "$VALIDATOR" ]; then
  python3 "$VALIDATOR" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$TESTDIR"
fi
