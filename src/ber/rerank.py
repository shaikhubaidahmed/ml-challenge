"""Blocking-stage reranker (Phase A, second pass).

The IDF-sum blocking score ranks true pairs poorly (recall@60 0.943 vs 0.968 within top-300).
A small LightGBM on four fast similarities re-scores the top-300 retrieved pairs per S1 and we
keep the top rerank_k. Trained on a sample of train S1 entities; applied per blocking chunk,
so its features must be computable per chunk (no cross-chunk statistics).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .config import preload_libomp

preload_libomp()
import lightgbm as lgb  # noqa: E402

RERANK_COLS = ["bscore", "n_keys", "rk", "name_ratio", "name_tset", "addr_tset", "addr_ratio", "addr_empty"]


def side_table(df: pl.DataFrame) -> pl.DataFrame:
    """idx == row position; only the strings the reranker needs."""
    return df.select("name_norm", "addr_norm").rechunk()


def rerank_features(pairs: pl.DataFrame, s1t: pl.DataFrame, poolt: pl.DataFrame) -> pl.DataFrame:
    A = s1t.select(pl.all().gather(pairs["s1_idx"]))
    B = poolt.select(pl.all().gather(pairs["pool_idx"]))
    na, nb, aa, ab = (A["name_norm"].to_list(), B["name_norm"].to_list(),
                      A["addr_norm"].to_list(), B["addr_norm"].to_list())

    def cp(x, y, sc):
        return process.cpdist(x, y, scorer=sc, workers=-1, dtype=np.float32)
    return pairs.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32).alias("rk"),
        pl.Series("name_ratio", cp(na, nb, fuzz.ratio)),
        pl.Series("name_tset", cp(na, nb, fuzz.token_set_ratio)),
        pl.Series("addr_tset", cp(aa, ab, fuzz.token_set_ratio)),
        pl.Series("addr_ratio", cp(aa, ab, fuzz.ratio)),
        (B["addr_norm"].str.len_chars() == 0).cast(pl.Float32).alias("addr_empty"),
    )


def train(d: pl.DataFrame, rounds=200) -> "lgb.Booster":
    params = dict(objective="binary", num_leaves=63, learning_rate=0.1, verbose=-1, num_threads=0, seed=42)
    return lgb.train(params, lgb.Dataset(d.select(RERANK_COLS).to_numpy(), d["y"].to_numpy()), rounds)


class Reranker:
    """Callable used by blocking: pairs(top-N by bscore) -> pairs(top-k by rerank score, with 'rr')."""

    def __init__(self, model_path, s1: pl.DataFrame, pool: pl.DataFrame, keep_k: int):
        self.model = lgb.Booster(model_file=str(model_path))
        self.s1t, self.poolt, self.keep_k = side_table(s1), side_table(pool), keep_k

    def __call__(self, pairs: pl.DataFrame) -> pl.DataFrame:
        f = rerank_features(pairs, self.s1t, self.poolt)
        rr = self.model.predict(f.select(RERANK_COLS).to_numpy(), num_threads=0).astype(np.float32)
        out = pairs.with_columns(pl.Series("rr", rr))
        return out.filter(pl.col("rr").rank("ordinal", descending=True).over("s1_idx") <= self.keep_k)
