#!/usr/bin/env bash
# Build <team>_submission.zip in the layout required by the problem statement.
# Usage: scripts/make_package.sh <team_name> <output_dir_with_tsvs>
set -euo pipefail
cd "$(dirname "$0")/.."
TEAM=${1:?team name}
OUTDIR=${2:?folder containing matching_results.tsv and candidate_pairs.tsv}
ROOT=../../package/${TEAM}_submission
rm -rf "$ROOT" && mkdir -p "$ROOT/output" "$ROOT/code/business_entity_resolution/src"
PKG=$ROOT/code/business_entity_resolution

cp "$OUTDIR/matching_results.tsv" "$OUTDIR/candidate_pairs.tsv" "$ROOT/output/"
cp -R src/ber "$PKG/src/ber"
cp -R scripts "$PKG/src/scripts"
mkdir -p "$PKG/src/configs" "$PKG/src/experiments"
cp configs/*.yaml "$PKG/src/configs/"
cp experiments/log.md "$PKG/src/experiments/log.md"
# package-level final config: data next to the zip root, work/ and output/ at the zip root
cat > "$PKG/src/configs/final.yaml" <<'YAML'
# Final submission config. Paths are relative to code/business_entity_resolution/.
paths:
  data_dir: ../../student_resource/dataset   # folder containing train/ and test/ (edit if elsewhere)
  work_dir: ../../work
  output_dir: ../../output
seed: 42
blocking:
  max_df: 500
  top_k: 300
  rerank_k: 50
  s1_chunk: 50000
validation:
  n_folds: 2
translit: true
YAML
sed -i '' 's#CFG=${1:-configs/v5.yaml}#CFG=${1:-configs/final.yaml}#; s#EXP=${2:-v5}#EXP=${2:-final}#' "$PKG/src/scripts/run_final.sh"
cp README.md requirements.txt "$PKG/"
cp Documentation_template.md "$ROOT/"
find "$ROOT" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$ROOT" -name ".DS_Store" -delete
(cd ../../package && rm -f "${TEAM}_submission.zip" && zip -qr "${TEAM}_submission.zip" "${TEAM}_submission")
ls -la ../../package/"${TEAM}_submission.zip"
(cd ../../package && unzip -l "${TEAM}_submission.zip" | grep -v "/$" | awk '{print $4}' | sed -n '1,200p')
