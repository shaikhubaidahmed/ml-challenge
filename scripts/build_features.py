"""Phase B: pair features for every stage-1 candidate, written as parquet shards.

Usage: python scripts/build_features.py --split train|test
Output: work/{split}_feats/part_XXX.parquet (train shards also carry label y and fold)
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo layout: scripts/ next to src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package layout: src/scripts/ next to src/ber/
from ber.config import load_config  # noqa: E402
from ber.data import build_split  # noqa: E402
from ber.features import add_group_features, pair_features, prepare_side  # noqa: E402
from ber.metrics import s1_subset_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--config", default=None)
ap.add_argument("--chunk_pairs", type=int, default=3_000_000)
ap.add_argument("--cands", default=None, help="override candidate parquet path")
ap.add_argument("--out", default=None, help="override output dir name under work/")
ap.add_argument("--s1_frac", type=float, default=1.0, help="keep only this fraction of S1 entities (train dev runs)")
ap.add_argument("--absent_s1", action="store_true",
                help="treat S1 entities outside --s1_frac as ABSENT: drop them before context features, so their "
                     "S2/S3 records become orphans (simulates test, where pool/S1 is 5.75 vs 4.68 in train)")
args = ap.parse_args()

cfg = load_config(args.config)
work = Path(cfg["paths"]["work_dir"])
s1, pool = build_split(cfg, args.split)
s1p, poolp = prepare_side(s1), prepare_side(pool)
gid = pool["gid"].to_numpy() if "gid" in pool.columns else None
n_folds = cfg["validation"]["n_folds"]
s1_fold = (s1["eid"].hash(seed=cfg["seed"]) % n_folds).cast(pl.Int8).rechunk()
del s1, pool

c = pl.read_parquet(args.cands or work / f"{args.split}_cands_stage1.parquet")
c = c.select("s1_idx", "pool_idx", "bscore", "n_keys", *(["rr"] if "rr" in c.columns else []))
if args.absent_s1:
    c = c.filter(s1_subset_mask(pl.col("s1_idx"), args.s1_frac))
c = add_group_features(c, "bscore", "b")  # context from the FULL candidate set (of present S1 entities)
if "rr" in c.columns:
    c = add_group_features(c, "rr", "r")
c = c.sort("s1_idx", "pool_idx")
if args.s1_frac < 1:
    c = c.filter(s1_subset_mask(pl.col("s1_idx"), args.s1_frac))
out = work / (args.out or f"{args.split}_feats")
shutil.rmtree(out, ignore_errors=True)
out.mkdir()
t = time.time()
for i, start in enumerate(range(0, c.height, args.chunk_pairs)):
    ch = c.slice(start, args.chunk_pairs)
    f = pair_features(ch.select("s1_idx", "pool_idx", "bscore", "n_keys"), s1p, poolp)
    f = pl.concat([f, ch.select(pl.selectors.starts_with("b_", "r_"), *(["rr"] if "rr" in ch.columns else []))],
                  how="horizontal")
    if gid is not None:
        g = pl.Series(gid).gather(ch["pool_idx"])
        f = f.with_columns((g == ch["s1_idx"].cast(pl.Int64)).cast(pl.Int8).alias("y"),
                           s1_fold.gather(ch["s1_idx"]).alias("fold"))
    f.write_parquet(out / f"part_{i:03d}.parquet", compression="zstd")
    print(f"  shard {i} rows={f.height:,} ({time.time()-t:.0f}s)", flush=True)
print(f"done {c.height:,} pairs in {time.time()-t:.0f}s")
