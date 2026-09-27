"""Train stage-1 and stage-2 models with 2-fold cross-fitting on train S1 entities and report
out-of-fold macro F0.5 (every train S1 entity is scored by a model that never saw it).

Usage: python scripts/train.py [--sample 0.3] [--exp NAME]
Writes: work/models/{exp}/stage{1,2}_fold{f}.txt, work/models/{exp}/oof.parquet, metrics.json
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
from ber.metrics import blocking_report, macro_f05, s1_subset_mask  # noqa: E402
from ber.decision import expected_f_select  # noqa: E402
from ber.model import (STAGE1_PARAMS, STAGE2_PARAMS, context_features, decide, feature_cols,  # noqa: E402
                       predict, train_lgb)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--exp", default="v1")
ap.add_argument("--sample", type=float, default=0.3, help="fraction of a fold's S1 entities used to fit stage 1")
ap.add_argument("--prune", type=float, default=0.01, help="stage-1 prob below which pairs are dropped")
ap.add_argument("--rounds", type=int, default=600)
ap.add_argument("--feats", default="train_feats")
ap.add_argument("--s1_frac", type=float, default=1.0, help="must match build_features --s1_frac")
ap.add_argument("--stage1_only", action="store_true")
ap.add_argument("--no_pool_ctx", action="store_true",
                help="drop p1-based pool-competition features (needed when only a subset of S1 is featurized)")
args = ap.parse_args()

cfg = load_config(args.config)
work = Path(cfg["paths"]["work_dir"])
mdir = work / "models" / args.exp
mdir.mkdir(parents=True, exist_ok=True)
shards = sorted((work / args.feats).glob("part_*.parquet"))
lf = pl.scan_parquet(shards)
cols = feature_cols(lf.head(1).collect())
print("stage-1 features:", len(cols), cols)
pool = pl.read_parquet(work / "train_pool.parquet", columns=["idx", "gid"])
n_s1 = pl.scan_parquet(work / "train_s1.parquet").select(pl.len()).collect().item()
all_s1 = pl.DataFrame({"i": np.arange(n_s1)}).filter(s1_subset_mask(pl.col("i"), args.s1_frac))["i"].to_numpy()
print(f"evaluating on {len(all_s1):,} S1 entities")
metrics = {"exp": args.exp, "args": vars(args)}
t0 = time.time()

# ---------------- stage 1: cross-fit ----------------
sub = (pl.col("s1_idx").hash(seed=1) % 1000) < int(args.sample * 1000)
models1 = {}
for f in (0, 1):
    tr = lf.filter((pl.col("fold") == f) & sub).collect()
    va = lf.filter((pl.col("fold") == 1 - f) & ((pl.col("s1_idx").hash(seed=2) % 100) == 0)).collect()
    print(f"[stage1 fold{f}] train rows={tr.height:,} pos={tr['y'].sum():,} valid rows={va.height:,}", flush=True)
    m = train_lgb(tr, va, cols, STAGE1_PARAMS, rounds=args.rounds)
    m.save_model(str(mdir / f"stage1_fold{f}.txt"))
    models1[f] = m
    del tr, va
print(f"stage1 trained ({time.time()-t0:.0f}s)", flush=True)

keep_raw = ["bscore", "b_rank_s1", "b_rank_pool", "name_tset", "name_ratio", "addr_tset", "addr_ratio",
            "jac_name_tok", "jac_addr_num", "deva_b", "src_b", "len_addr_b"]
oof = []
for sh in shards:
    d = pl.read_parquet(sh)
    p = np.empty(d.height, dtype=np.float32)
    fold = d["fold"].to_numpy()
    for f in (0, 1):
        m = fold == f
        if m.any():
            p[m] = predict(models1[1 - f], d.filter(pl.Series(m)), cols)
    # learned pruning applied per shard to bound memory; pairs below prune are never matched
    oof.append(d.select("s1_idx", "pool_idx", "y", "fold", *keep_raw).with_columns(pl.Series("p1", p))
               .filter(pl.col("p1") >= args.prune))
oof = pl.concat(oof)
print(f"stage1 OOF predicted rows={oof.height:,} ({time.time()-t0:.0f}s)", flush=True)

# stage-1-only decision (reference)
res1 = {}
for thr in np.round(np.arange(0.2, 0.96, 0.05), 2):
    for excl in (True, False):
        r = macro_f05(decide(oof, float(thr), "p1", exclusive=excl), all_s1, pool)
        res1[(float(thr), excl)] = r
        print(f"[stage1] thr={thr:.2f} excl={excl} f05={r['f05']:.5f} P={r['precision_macro']:.4f} "
              f"R={r['recall_macro']:.4f} single={r['singleton_acc']:.4f}", flush=True)
(bt1, be1), br1 = max(res1.items(), key=lambda kv: kv[1]["f05"])
metrics["stage1_only"] = {"thr": bt1, "exclusive": be1, **br1}
print("stage1-only best:", metrics["stage1_only"], flush=True)

# ---------------- learned pruning -> final candidate set ----------------
cand = oof.filter(pl.col("p1") >= args.prune)
metrics["candidates"] = blocking_report(cand, all_s1, pool)
metrics["stage1_blocking"] = blocking_report(lf.select("s1_idx", "pool_idx").collect(), all_s1, pool)
print("stage1 blocking:", metrics["stage1_blocking"], "\npruned candidates:", metrics["candidates"], flush=True)
del oof
if args.stage1_only:
    (mdir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    sys.exit(0)

# ---------------- stage 2: cross-fit on context features ----------------
cand = context_features(cand, "p1")
cols2 = feature_cols(cand)
if args.no_pool_ctx:
    cols2 = [c for c in cols2 if not (c.startswith("p1_") and ("pool" in c or "best_other" in c))]
print("stage-2 features:", len(cols2), cols2)
p2 = np.empty(cand.height, dtype=np.float32)
fold = cand["fold"].to_numpy()
for f in (0, 1):
    tr = cand.filter(pl.Series(fold == f))
    va = cand.filter(pl.Series(fold == 1 - f)).sample(fraction=0.05, seed=3)
    m = train_lgb(tr, va, cols2, STAGE2_PARAMS, rounds=args.rounds)
    m.save_model(str(mdir / f"stage2_fold{f}.txt"))
    msk = fold == 1 - f
    p2[msk] = predict(m, cand.filter(pl.Series(msk)), cols2)
cand = cand.with_columns(pl.Series("p2", p2))
cand.select("s1_idx", "pool_idx", "y", "fold", "p1", "p2").write_parquet(mdir / "oof.parquet")

# ---------------- threshold search (per fold, to see stability) ----------------
res = {}
for thr in np.round(np.arange(0.2, 0.96, 0.05), 2):
    for excl in (True, False):
        pred = decide(cand, thr, "p2", exclusive=excl)
        r = macro_f05(pred, all_s1, pool)
        res[(float(thr), excl)] = r
        print(f"thr={thr:.2f} excl={excl} f05={r['f05']:.5f} P={r['precision_macro']:.4f} R={r['recall_macro']:.4f} "
              f"single={r['singleton_acc']:.4f}", flush=True)
(bt, be), br = max(res.items(), key=lambda kv: kv[1]["f05"])
per_fold = {}
s1_fold = pl.read_parquet(work / "train_s1.parquet", columns=["eid"])["eid"].hash(seed=cfg["seed"]) % cfg["validation"]["n_folds"]
for f in (0, 1):
    ids = all_s1[s1_fold.to_numpy()[all_s1] == f]
    per_fold[f] = macro_f05(decide(cand, bt, "p2", exclusive=be), ids, pool)["f05"]
metrics["stage2"] = {"thr": bt, "exclusive": be, **br, "per_fold_f05": per_fold, "rule": "threshold"}
# alternative decision rule: per-entity expected-F0.5 after exclusive assignment
ex = cand.filter(pl.col("p2") == pl.col("p2").max().over("pool_idx")).unique("pool_idx", keep="first")
best_ef = None
for extra in (0.0, 0.1, 0.3):
    r = macro_f05(expected_f_select(ex, "p2", extra=extra), all_s1, pool)
    print(f"[expF] extra={extra} f05={r['f05']:.5f} P={r['precision_macro']:.4f} R={r['recall_macro']:.4f} "
          f"single={r['singleton_acc']:.4f}", flush=True)
    if best_ef is None or r["f05"] > best_ef[1]["f05"]:
        best_ef = (extra, r)
metrics["stage2_expected_f"] = {"extra": best_ef[0], **best_ef[1]}
if best_ef[1]["f05"] > br["f05"]:
    metrics["stage2"]["rule"] = "expected_f"
    metrics["stage2"]["extra"] = best_ef[0]
metrics["runtime_s"] = time.time() - t0
print(json.dumps(metrics, indent=1, default=str))
(mdir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
