"""Stage-1 blocking for a whole split. Writes work/{split}_cands_stage1.parquet.

Usage: python scripts/run_blocking.py --split train|test
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo layout: scripts/ next to src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package layout: src/scripts/ next to src/ber/
from ber.blocking import KEY_TYPES, add_token_hashes, generate_candidates  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.data import build_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
b = cfg["blocking"]
s1, pool = build_split(cfg, args.split)
cols = ["idx", "country_norm", "name_tok", "addr_tok"]
s1h, poolh = add_token_hashes(s1.select(cols)), add_token_hashes(pool.select(cols))
reranker = None
if b.get("rerank_k"):
    from ber.rerank import Reranker  # noqa: E402
    reranker = Reranker(Path(cfg["paths"]["work_dir"]) / "reranker.txt", s1, pool, b["rerank_k"])
    print(f"reranking top {b['top_k']} -> keep {b['rerank_k']}")
del s1, pool
caps = {t: b["max_df"] for t in KEY_TYPES}
t = time.time()
c = generate_candidates(s1h, poolh, max_df=caps, top_k=b["top_k"], s1_chunk=b["s1_chunk"], reranker=reranker)
out = Path(cfg["paths"]["work_dir"]) / f"{args.split}_cands_stage1.parquet"
c.sort("s1_idx", "pool_idx").write_parquet(out)
print(f"wrote {out} pairs={c.height:,} s1_with_cands={c['s1_idx'].n_unique():,} time={time.time()-t:.0f}s")
