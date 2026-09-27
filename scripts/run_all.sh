#!/usr/bin/env bash
# End-to-end reproduction: data -> blocking -> features -> models -> submission files -> validation.
# Usage: scripts/run_all.sh [config] [exp_name]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/v2.yaml}
EXP=${2:-v2}
PY=${PY:-../../.venv/bin/python}

$PY scripts/run_blocking.py --config "$CFG" --split train      # (v3+: needs reranker.txt from train_reranker.py) also builds normalized caches + transliteration map
$PY scripts/run_blocking.py --config "$CFG" --split test
$PY scripts/build_features.py --config "$CFG" --split train
$PY scripts/build_features.py --config "$CFG" --split test
$PY scripts/train.py --config "$CFG" --exp "$EXP"
$PY scripts/predict.py --config "$CFG" --exp "$EXP"
OUT=$($PY -c "import sys; sys.path.insert(0,'src'); from ber.config import load_config; print(load_config('$CFG')['paths']['output_dir'])")
python3 ../../student_resource/utils/validate_submission.py \
    --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
    --test-dir ../../student_resource/dataset/test
