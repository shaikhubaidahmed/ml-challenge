"""Score an existing experiment's stage-1 models (out-of-fold by S1 fold) on another feature set.

Used to measure distribution shift: e.g. v4 models on the absent-S1 (test-like) train features.
Usage: python scripts/eval_models.py --config configs/v4.yaml --exp v4 --feats train_feats_abs_nev --s1_frac 0.81
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo layout: scripts/ next to src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package layout: src/scripts/ next to src/ber/
from ber.config import load_config  # noqa: E402
from ber.metrics import macro_f05, s1_subset_mask  # noqa: E402
from ber.model import decide, lgb, predict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--exp", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--s1_frac", type=float, default=1.0)
args = ap.parse_args()

cfg = load_config(args.config)
work = Path(cfg["paths"]["work_dir"])
mdir = work / "models" / args.exp
m1 = {f: lgb.Booster(model_file=str(mdir / f"stage1_fold{f}.txt")) for f in (0, 1)}
cols = m1[0].feature_name()
out = []
for sh in sorted((work / args.feats).glob("part_*.parquet")):
    d = pl.read_parquet(sh, columns=["s1_idx", "pool_idx", "fold", *cols])
    p = np.empty(d.height, dtype=np.float32)
    fold = d["fold"].to_numpy()
    for f in (0, 1):
        msk = fold == f
        if msk.any():
            p[msk] = predict(m1[1 - f], d.filter(pl.Series(msk)), cols)
    out.append(d.select("s1_idx", "pool_idx").with_columns(pl.Series("p1", p)).filter(pl.col("p1") >= 0.01))
sc = pl.concat(out)
pool = pl.read_parquet(work / "train_pool.parquet", columns=["idx", "gid"])
n = pl.scan_parquet(work / "train_s1.parquet").select(pl.len()).collect().item()
ids = pl.DataFrame({"i": np.arange(n)}).filter(s1_subset_mask(pl.col("i"), args.s1_frac))["i"].to_numpy()
for thr in (0.6, 0.7, 0.8, 0.9):
    r = macro_f05(decide(sc, thr, "p1", True), ids, pool)
    print(f"[{args.exp} stage1 on {args.feats}] thr={thr} f05={r['f05']:.5f} P={r['precision_macro']:.4f} "
          f"R={r['recall_macro']:.4f} single={r['singleton_acc']:.4f}", flush=True)
