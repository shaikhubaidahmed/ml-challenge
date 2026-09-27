"""Candidate generation (Phase A).

Inverted-index retrieval, run separately per country (100% of train positives share country;
country is an open-set string key, so unseen labels such as France are handled identically).

Each record yields hashed integer keys of several types:
  n1  single name token                a1  single address token
  nn  unordered pair of name tokens    ab  adjacent address-token bigram
Tokens are 32-bit hashes (name and address use different seeds); pair keys pack two hashes
into a u64. Keys whose pool document frequency exceeds a per-type cap are dropped. A pair
(S1, pool) is a candidate if it shares >= 1 surviving key; its blocking score is the
IDF-weighted sum of shared keys. The top_k pool records per S1 entity are kept.
"""
import math
import time

import numpy as np
import polars as pl

KEY_TYPES = ("n1", "a1", "nn", "ab")
_TYPE_ID = {t: i + 1 for i, t in enumerate(KEY_TYPES)}


def add_token_hashes(df: pl.DataFrame) -> pl.DataFrame:
    """UInt32 hash lists for name/address tokens (distinct seeds so fields never collide)."""
    def h(col, seed):
        return pl.col(col).list.eval((pl.element().hash(seed=seed) & 0xFFFFFFFF).cast(pl.UInt32))
    return df.with_columns(h("name_tok", 11).alias("name_h"), h("addr_tok", 23).alias("addr_h"))


def record_keys(df: pl.DataFrame, types=KEY_TYPES) -> pl.DataFrame:
    """Explode records (needs idx, name_h, addr_h) into unique (idx, t, key) rows."""
    parts = []
    base = df.select(pl.col("idx").cast(pl.UInt32), "name_h", "addr_h")
    if "n1" in types or "nn" in types:
        n = base.select("idx", pl.col("name_h").list.unique().alias("h")).explode("h").drop_nulls()
        if "n1" in types:
            parts.append(n.select("idx", pl.lit(_TYPE_ID["n1"], pl.UInt8).alias("t"),
                                  (pl.col("h").cast(pl.UInt64) + (1 << 32)).alias("key")))
        if "nn" in types:
            nn = n.join(n, on="idx", suffix="_b").filter(pl.col("h") < pl.col("h_b"))
            parts.append(nn.select("idx", pl.lit(_TYPE_ID["nn"], pl.UInt8).alias("t"),
                                   (pl.col("h").cast(pl.UInt64) * (1 << 32) + pl.col("h_b").cast(pl.UInt64)).alias("key")))
    if "a1" in types:
        a = base.select("idx", pl.col("addr_h").alias("h")).explode("h").drop_nulls()
        parts.append(a.select("idx", pl.lit(_TYPE_ID["a1"], pl.UInt8).alias("t"),
                              (pl.col("h").cast(pl.UInt64) + (2 << 32)).alias("key")))
    if "ab" in types:
        ab = base.select("idx", pl.col("addr_h").list.slice(0, pl.col("addr_h").list.len() - 1).alias("h1"),
                         pl.col("addr_h").list.slice(1).alias("h2")).explode(["h1", "h2"]).drop_nulls()
        # swap high/low halves so address bigrams never share the name-pair key space
        parts.append(ab.select("idx", pl.lit(_TYPE_ID["ab"], pl.UInt8).alias("t"),
                               (pl.col("h2").cast(pl.UInt64) * (1 << 32) + pl.col("h1").cast(pl.UInt64)).xor(pl.lit(0x5bd1e9955bd1e995, pl.UInt64)).alias("key")))
    return pl.concat(parts).unique(["idx", "key"])


def build_index(pool_c: pl.DataFrame, max_df: dict):
    """Postings (key, pool_idx, idf) for one country's pool, dropping keys above the per-type cap."""
    pk = record_keys(pool_c)
    df = pk.group_by("key", "t").agg(pl.len().alias("df"))
    caps = pl.DataFrame({"t": [_TYPE_ID[t] for t in max_df], "cap": list(max_df.values())},
                        schema={"t": pl.UInt8, "cap": pl.Int64})
    df = df.join(caps, on="t").filter(pl.col("df") <= pl.col("cap"))
    logn = math.log(max(pool_c.height, 2))
    df = df.with_columns((logn - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32).alias("idf"))
    return pk.join(df.select("key", "idf"), on="key").select("key", pl.col("idx").alias("pool_idx"), "idf")


def generate_candidates(s1: pl.DataFrame, pool: pl.DataFrame, max_df: dict, top_k=50, s1_chunk=100_000,
                        verbose=True, reranker=None) -> pl.DataFrame:
    """Return candidate pairs (s1_idx, pool_idx, bscore, n_keys), <= top_k per S1, same country only."""
    t0 = time.time()
    out = []
    for country in s1["country_norm"].unique().sort().to_list():
        s1_c = s1.filter(pl.col("country_norm") == country)
        pool_c = pool.filter(pl.col("country_norm") == country)
        if pool_c.height == 0:
            continue
        idx = build_index(pool_c, max_df)
        if verbose:
            print(f"  [{country}] s1={s1_c.height:,} pool={pool_c.height:,} postings={idx.height:,} ({time.time()-t0:.0f}s)")
        for start in range(0, s1_c.height, s1_chunk):
            s1k = record_keys(s1_c.slice(start, s1_chunk))
            pairs = (s1k.join(idx, on="key")
                     .group_by(pl.col("idx").alias("s1_idx"), "pool_idx")
                     .agg(pl.col("idf").sum().alias("bscore"), pl.len().cast(pl.UInt16).alias("n_keys")))
            pairs = pairs.filter(pl.col("bscore").rank("ordinal", descending=True).over("s1_idx") <= top_k)
            if reranker is not None:  # second pass: rerank top_k by cheap similarities, keep reranker.keep_k
                pairs = reranker(pairs)
            out.append(pairs)
        if verbose:
            print(f"  [{country}] done ({time.time()-t0:.0f}s)")
        del idx
    return pl.concat(out)
