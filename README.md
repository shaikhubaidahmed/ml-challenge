# Business Entity Resolution — Amazon ML Challenge 2026

Given business records from three sources, find for every Source 1 entity all matching Source 2 / Source 3
records. Output: `output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv` (the exact
candidate set scored by the final model). Methodology: see `Documentation_template.md` at the zip root.

Only the provided challenge data is used. No external lookups, APIs, geocoders, business databases,
or pretrained language models. All models are LightGBM (MIT licence).

## Pipeline

```
TSV → normalization (+ Indic-script → Latin dictionary learned from train pairs)
    → blocking per country: compound-key inverted index, top-300 by IDF score
    → learned reranker keeps top-50 per S1
    → pair features (string similarity, name-evidence IDF, competition context)
    → stage-1 LightGBM (cross-fitted) → learned pruning p1 ≥ 0.01  (= candidate_pairs.tsv)
    → stage-2 LightGBM on stage-1 context
    → exclusive assignment (each S2/S3 record → at most one S1) + per-entity expected-F0.5 selection
    → matching_results.tsv → official validator
```

## Environment

Python 3.12. Tested on macOS (Apple M5, 10 cores, 16 GB RAM); peak memory ≈ 8 GB.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

macOS only: LightGBM needs `libomp` (`brew install libomp`). `ber.config.preload_libomp()` also finds it via
`LIBOMP_PATH` or an Anaconda install.

## Data

Edit `src/configs/final.yaml` → `paths.data_dir` to the folder that contains `train/` and `test/`
(default: `../../student_resource/dataset`, relative to this folder). Intermediate files go to
`paths.work_dir` (~40 GB free disk recommended), outputs to `paths.output_dir`.

## Reproduce end to end

From this folder (`code/business_entity_resolution/`):

```bash
PY=python src/scripts/run_final.sh            # configs/final.yaml, experiment "final"
```

Steps executed (each is also runnable on its own; all take `--config`):

| # | Script (under `src/scripts/`) | Output | Time* |
|---|---|---|---|
| 1 | `train_reranker.py` | normalized caches, `translit.json`, `reranker.txt` | 5 min |
| 2 | `run_blocking.py --split train/test` | `*_cands_stage1.parquet` | 33 + 26 min |
| 3 | `build_features.py` (train with `--s1_frac 0.81 --absent_s1`) | feature shards | 9 + 8 min |
| 4 | `add_name_features.py` | + name-evidence features | 5 + 3 min |
| 5 | `train.py` | cross-fitted models, OOF metrics, decision rule (`models/<exp>/metrics.json`) | 35 min |
| 6 | `predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | 40 min |
| 7 | official `validate_submission.py` | PASS | 1 min |

*measured on the machine above.

`--absent_s1` (step 3): the test set has 5.75 S2/S3 records per S1 versus 4.68 in train, i.e. many S2/S3
records whose entity is not in Source 1. Training treats 19% of train S1 entities as absent so their records
become such orphans, making training and validation look like test.

## Source layout (`src/`)

| Path | Purpose |
|---|---|
| `ber/data.py` | TSV loading (tab, quoting disabled), parquet caches, ground truth → labels |
| `ber/normalize.py` | cleaning, accent folding, field-specific abbreviations, Indic tokens → Latin |
| `ber/translit.py` | transliteration dictionary learned only from training pairs |
| `ber/blocking.py` | per-country inverted index over hashed single + compound keys, IDF top-K |
| `ber/rerank.py` | second-pass LightGBM reranker for blocking |
| `ber/features.py` | rapidfuzz similarities, token/number Jaccard, blocking-context features |
| `ber/features_name.py` | IDF name-evidence and name-ambiguity features |
| `ber/model.py` | LightGBM stages, stage-2 context features, threshold decision |
| `ber/decision.py` | per-entity expected-F0.5 selection |
| `ber/metrics.py` | official macro F0.5 per Source 1 entity, blocking recall ceiling |
| `ber/submission.py` | TSV writers (one row per S1, comma-joined IDs, no quoting) |
| `scripts/` | pipeline steps, `run_final.sh`, analysis helpers (`eval_blocking.py`, `eval_models.py`) |
| `configs/` | `final.yaml` (submission) and per-version configs used during development |
| `experiments/log.md` | experiment log with every measured result and the submission ledger |
