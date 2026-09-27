"""Phases C-F on the test split: score candidates, decide matches, write submission files.

Usage: python scripts/predict.py --exp v1 [--stage 1|2]
Reads thresholds from work/models/{exp}/metrics.json. Writes to paths.output_dir:
  matching_results.tsv, candidate_pairs.tsv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo layout: scripts/ next to src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package layout: src/scripts/ next to src/ber/
from ber.config import load_config  # noqa: E402
from ber.decision import expected_f_select  # noqa: E402
from ber.model import context_features, decide, lgb, predict  # noqa: E402
from ber.submission import write_id_lists  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--exp", default="v1")
ap.add_argument("--stage", type=int, default=2)
ap.add_argument("--feats", default="test_feats")
args = ap.parse_args()

cfg = load_config(args.config)
work = Path(cfg["paths"]["work_dir"])
mdir = work / "models" / args.exp
metrics = json.loads((mdir / "metrics.json").read_text())
out_dir = Path(cfg["paths"]["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()

m1 = [lgb.Booster(model_file=str(mdir / f"stage1_fold{f}.txt")) for f in (0, 1)]
cols1 = m1[0].feature_name()
prune = metrics["args"]["prune"]
p1_cache = mdir / f"{args.feats}_p1_pruned.parquet"
if p1_cache.exists():
    scored = pl.read_parquet(p1_cache)
    print(f"  loaded cached stage-1 scores {p1_cache.name}")
else:
    scored = []
    for sh in sorted((work / args.feats).glob("part_*.parquet")):
        d = pl.read_parquet(sh)
        p = np.mean([predict(m, d, cols1) for m in m1], axis=0).astype(np.float32)
        # learned pruning: only pairs with p1 >= prune reach the final model (== candidate_pairs.tsv)
        d = d.select("s1_idx", "pool_idx", *cols1).with_columns(pl.Series("p1", p))
        scored.append(d.filter(pl.col("p1") >= prune))
        print(f"  scored {sh.name} ({time.time()-t0:.0f}s)", flush=True)
    scored = pl.concat(scored)
    scored.write_parquet(p1_cache)
assert scored["p1"].min() >= np.float32(prune)  # float32 comparison, same as the filter

if args.stage == 1:
    cfg1 = metrics["stage1_only"]
    cand = scored
    pred = decide(cand, cfg1["thr"], "p1", exclusive=cfg1.get("exclusive", True))
else:
    cand = context_features(scored, "p1")
    m2 = [lgb.Booster(model_file=str(mdir / f"stage2_fold{f}.txt")) for f in (0, 1)]
    cols2 = m2[0].feature_name()
    cand = cand.with_columns(pl.Series("p2", np.mean([predict(m, cand, cols2) for m in m2], axis=0)))
    s2 = metrics["stage2"]
    if s2.get("rule") == "expected_f":
        ex = cand.filter(pl.col("p2") == pl.col("p2").max().over("pool_idx")).unique("pool_idx", keep="first")
        pred = expected_f_select(ex, "p2", extra=s2["extra"])
    else:
        pred = decide(cand, s2["thr"], "p2", exclusive=s2["exclusive"])

s1 = pl.read_parquet(work / "test_s1.parquet", columns=["idx", "eid"])
pool = pl.read_parquet(work / "test_pool.parquet", columns=["idx", "eid"])
write_id_lists(cand.select("s1_idx", "pool_idx"), s1, pool, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
write_id_lists(pred, s1, pool, out_dir / "matching_results.tsv", "matched_entity_ids")
n_match = pred.group_by("s1_idx").len()
print(f"candidates={cand.height:,} matches={pred.height:,} s1_with_match={n_match.height:,}/{s1.height:,} "
      f"({time.time()-t0:.0f}s)")
