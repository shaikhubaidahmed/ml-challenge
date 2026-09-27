"""Append name-evidence features (src/ber/features_name.py) to existing feature shards.

Usage: python scripts/add_name_features.py --config configs/v3.yaml --split train|test [--src train_feats --dst train_feats_nev]
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
from ber.features_name import name_evidence_features, name_tables  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--src", default=None)
ap.add_argument("--dst", default=None)
ap.add_argument("--present_s1_frac", type=float, default=1.0,
                help="S1 name counts use only S1 entities inside this subset (must match build_features --absent_s1)")
args = ap.parse_args()

cfg = load_config(args.config)
work = Path(cfg["paths"]["work_dir"])
src = work / (args.src or f"{args.split}_feats")
dst = work / (args.dst or f"{args.split}_feats_nev")
s1, pool = build_split(cfg, args.split)
present = None
if args.present_s1_frac < 1:
    from ber.metrics import s1_subset_mask  # noqa: E402
    present = s1.select(s1_subset_mask(pl.col("idx"), args.present_s1_frac))
    present = present.to_series()
s1n, pooln, idf = name_tables(s1, pool, present=present)
del s1, pool
shutil.rmtree(dst, ignore_errors=True)
dst.mkdir()
t = time.time()
for sh in sorted(src.glob("part_*.parquet")):
    d = pl.read_parquet(sh)
    extra = name_evidence_features(d.select("s1_idx", "pool_idx"), s1n, pooln, idf)
    pl.concat([d, extra], how="horizontal").write_parquet(dst / sh.name, compression="zstd")
    print(f"  {sh.name} ({time.time()-t:.0f}s)", flush=True)
print(f"done {args.split} in {time.time()-t:.0f}s")
