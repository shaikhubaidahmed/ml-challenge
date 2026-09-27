"""Measure blocking quality on a sample of train S1 entities against the FULL train pool.

Usage: python scripts/eval_blocking.py --n 100000 --max_df 400 --top_k 60
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
from ber.blocking import KEY_TYPES, add_token_hashes, generate_candidates  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.data import build_split  # noqa: E402
from ber.metrics import blocking_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--max_df", type=int, default=500)
ap.add_argument("--caps", type=str, default="", help="per-type caps e.g. n1=500,a1=500,nn=500,ab=500")
ap.add_argument("--top_k", type=int, default=60)
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
s1, pool = build_split(cfg, "train")
rng = np.random.default_rng(cfg["seed"])
sample = np.sort(rng.choice(s1.height, args.n, replace=False))
s1s = add_token_hashes(s1[sample.tolist()])
pool_h = add_token_hashes(pool.select("idx", "country_norm", "name_tok", "addr_tok"))
caps = {t: args.max_df for t in KEY_TYPES}
for kv in filter(None, args.caps.split(",")):
    k, v = kv.split("="); caps[k] = int(v)
print("caps", caps)
t = time.time()
c = generate_candidates(s1s, pool_h, max_df=caps, top_k=args.top_k, s1_chunk=50_000)
print(f"blocking time {time.time()-t:.0f}s")
c = c.with_columns(pl.col("bscore").rank("ordinal", descending=True).over("s1_idx").alias("rk"))
for k in sorted({10, 20, 40, 60, 80, 100, 150, 200, args.top_k}):
    if k <= args.top_k:
        r = blocking_report(c.filter(pl.col("rk") <= k), sample, pool)
        print(f"top_k={k}: " + json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in r.items()}))
out = Path(cfg["paths"]["work_dir"]) / "blocking_eval_cands.parquet"
c.write_parquet(out)
