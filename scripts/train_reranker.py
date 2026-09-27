"""Train the blocking reranker on a sample of train S1 entities (against the FULL train pool).

Usage: python scripts/train_reranker.py --config configs/v3.yaml [--n 60000]
Writes work_dir/reranker.txt and prints holdout recall@K (bscore order vs reranked order).
"""
import argparse
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
from ber.rerank import rerank_features, side_table, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n", type=int, default=60_000)
args = ap.parse_args()

cfg = load_config(args.config)
b = cfg["blocking"]
s1, pool = build_split(cfg, "train")
rng = np.random.default_rng(cfg["seed"] + 7)
sample = np.sort(rng.choice(s1.height, args.n, replace=False))
cols = ["idx", "country_norm", "name_tok", "addr_tok"]
t = time.time()
c = generate_candidates(add_token_hashes(s1[sample.tolist()].select(cols)), add_token_hashes(pool.select(cols)),
                        max_df={k: b["max_df"] for k in KEY_TYPES}, top_k=b["top_k"], s1_chunk=args.n)
d = rerank_features(c, side_table(s1), side_table(pool))
gid = pool["gid"].gather(d["pool_idx"])
d = d.with_columns((gid == d["s1_idx"].cast(pl.Int64)).alias("y"))
print(f"pairs={d.height:,} positives={d['y'].sum():,} ({time.time()-t:.0f}s)")

half = (pl.col("s1_idx") % 2) == 0
m = train(d.filter(half))
te = d.filter(~half)
from ber.rerank import RERANK_COLS  # noqa: E402
te = te.with_columns(pl.Series("rr", m.predict(te.select(RERANK_COLS).to_numpy())))
te = te.with_columns(pl.col("rr").rank("ordinal", descending=True).over("s1_idx").alias("rrk"))
tot = pool.filter(pl.col("gid").is_in(te["s1_idx"].unique().cast(pl.Int64).implode())).height
for k in (20, 40, 50, 60, 100):
    print(f"holdout K={k}: bscore recall={te.filter((pl.col('rk') <= k) & pl.col('y')).height / tot:.4f} "
          f"reranked recall={te.filter((pl.col('rrk') <= k) & pl.col('y')).height / tot:.4f}")
final = train(d)  # final reranker uses all sampled entities
out = Path(cfg["paths"]["work_dir"]) / "reranker.txt"
final.save_model(str(out))
print(f"saved {out}")
