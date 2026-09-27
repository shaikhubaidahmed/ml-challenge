"""Pair features (Phase B). Country-agnostic string similarities; no country one-hot.

Input: candidate pairs (s1_idx, pool_idx, bscore, n_keys) plus the normalized s1/pool tables,
whose idx equals row position. Output: one float32 feature row per pair.
"""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

FIELDS = ["name_norm", "addr_norm", "name_tok", "addr_tok", "name_deva"]


def _num_tokens(col):
    return pl.col(col).list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")))


def prepare_side(df: pl.DataFrame) -> pl.DataFrame:
    """Columns needed for features, with idx == row position."""
    return df.select(
        "idx", "name_norm", "addr_norm", "name_tok", "addr_tok", "name_deva",
        pl.col("name_norm").str.replace_all(" ", "").alias("name_ns"),
        _num_tokens("addr_tok").alias("addr_num"),
        pl.col("src") if "src" in df.columns else pl.lit(1, pl.UInt8).alias("src"),
    ).rechunk()


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _jacc(a: str, b: str) -> list:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return [
        (inter / pl.max_horizontal(union, 1)).cast(pl.Float32).alias(f"jac_{a[:-2]}"),
        inter.cast(pl.Float32).alias(f"ncommon_{a[:-2]}"),
    ]


def pair_features(pairs: pl.DataFrame, s1p: pl.DataFrame, poolp: pl.DataFrame) -> pl.DataFrame:
    """Compute features for a chunk of candidate pairs."""
    A = s1p.select(pl.all().gather(pairs["s1_idx"])).drop("idx", "src")
    B = poolp.select(pl.all().gather(pairs["pool_idx"])).drop("idx")
    d = pl.concat([A.rename({c: c + "_a" for c in A.columns}), B.rename({c: c + "_b" for c in B.columns})],
                  how="horizontal")
    feats = d.select(
        *_jacc("name_tok_a", "name_tok_b"),
        *_jacc("addr_tok_a", "addr_tok_b"),
        *_jacc("addr_num_a", "addr_num_b"),
        pl.col("name_tok_a").list.first().eq(pl.col("name_tok_b").list.first()).cast(pl.Float32).alias("first_tok_eq"),
        pl.col("name_norm_a").str.len_chars().cast(pl.Float32).alias("len_name_a"),
        pl.col("name_norm_b").str.len_chars().cast(pl.Float32).alias("len_name_b"),
        pl.col("addr_norm_a").str.len_chars().cast(pl.Float32).alias("len_addr_a"),
        pl.col("addr_norm_b").str.len_chars().cast(pl.Float32).alias("len_addr_b"),
        pl.col("addr_num_a").list.len().cast(pl.Float32).alias("nnum_a"),
        pl.col("addr_num_b").list.len().cast(pl.Float32).alias("nnum_b"),
        (pl.col("addr_num_a").list.set_difference(pl.col("addr_num_b")).list.len() == 0).cast(pl.Float32).alias("num_a_subset_b"),
        pl.col("name_deva_b").cast(pl.Float32).alias("deva_b"),
        pl.col("src_b").cast(pl.Float32).alias("src_b"),
    )
    na, nb = d["name_norm_a"].to_list(), d["name_norm_b"].to_list()
    aa, ab = d["addr_norm_a"].to_list(), d["addr_norm_b"].to_list()
    nsa, nsb = d["name_ns_a"].to_list(), d["name_ns_b"].to_list()
    f = {
        "name_ratio": _cp(na, nb, fuzz.ratio),
        "name_tsort": _cp(na, nb, fuzz.token_sort_ratio),
        "name_tset": _cp(na, nb, fuzz.token_set_ratio),
        "name_partial": _cp(na, nb, fuzz.partial_ratio),
        "name_jw": _cp(na, nb, distance.JaroWinkler.normalized_similarity),
        "name_ns_ratio": _cp(nsa, nsb, fuzz.ratio),
        "name_ns_partial": _cp(nsa, nsb, fuzz.partial_ratio),
        "addr_ratio": _cp(aa, ab, fuzz.ratio),
        "addr_tsort": _cp(aa, ab, fuzz.token_sort_ratio),
        "addr_tset": _cp(aa, ab, fuzz.token_set_ratio),
        "addr_partial": _cp(aa, ab, fuzz.partial_ratio),
    }
    blk = pairs.select(pl.col("bscore").cast(pl.Float32), pl.col("n_keys").cast(pl.Float32))
    return pl.concat([pairs.select("s1_idx", "pool_idx"), blk, feats, pl.DataFrame(f)], how="horizontal")


def add_group_features(df: pl.DataFrame, score_col="bscore", prefix="b") -> pl.DataFrame:
    """Context features: how this pair ranks among the S1 entity's candidates and among the
    S1 entities competing for the same pool record (each pool record matches <= 1 S1)."""
    s = pl.col(score_col)
    return df.with_columns(
        s.rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32).alias(f"{prefix}_rank_s1"),
        s.rank("ordinal", descending=True).over("pool_idx").cast(pl.Float32).alias(f"{prefix}_rank_pool"),
        (s - s.max().over("s1_idx")).cast(pl.Float32).alias(f"{prefix}_gap_s1max"),
        (s - s.max().over("pool_idx")).cast(pl.Float32).alias(f"{prefix}_gap_poolmax"),
        pl.len().over("pool_idx").cast(pl.Float32).alias(f"{prefix}_n_s1_for_pool"),
    )
