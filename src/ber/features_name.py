"""Name-evidence features (v4): how distinctive the agreeing/disagreeing name tokens are.

Motivation (v2 error analysis): 87% of false merges are decoy records (same address, one altered
name token); 32% of in-candidate misses have no address, so the name is the only evidence.
All statistics are computed from the split's own records (unsupervised, no labels, no external data):
  - n_s1_same_name / n_pool_same_name: how many S1 / pool records in the same country carry
    exactly this normalized name (name ambiguity).
  - IDF (over S1 + pool name tokens, per country) of shared tokens, S1-only tokens, pool-only tokens.
  - leftover_ratio: char similarity of the unshared tokens (typo -> high, different word -> low).
"""
import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def name_tables(s1: pl.DataFrame, pool: pl.DataFrame, present: pl.Series | None = None):
    """Per-record name statistics (idx == row position) and a per-country token IDF table.
    present: optional boolean mask over s1 rows; only present S1 entities are counted (absent-S1 simulation)."""
    key = ["country_norm", "name_norm"]
    s1_present = s1 if present is None else s1.filter(present)
    c1 = s1_present.group_by(key).agg(pl.len().alias("n_s1_same_name"))
    c2 = pool.group_by(key).agg(pl.len().alias("n_pool_same_name"))

    def side(df):
        return (df.select("idx", *key, "name_tok").join(c1, on=key, how="left").join(c2, on=key, how="left")
                .sort("idx").select("name_tok", "country_norm",
                                    pl.col("n_s1_same_name").fill_null(0).cast(pl.Float32),
                                    pl.col("n_pool_same_name").fill_null(0).cast(pl.Float32)).rechunk())
    both = pl.concat([s1_present.select("country_norm", "name_tok"), pool.select("country_norm", "name_tok")])
    n_c = both.group_by("country_norm").len().rename({"len": "N"})
    idf = (both.select("country_norm", pl.col("name_tok").list.unique().alias("tok")).explode("tok")
           .drop_nulls().group_by("country_norm", "tok").len().join(n_c, on="country_norm")
           .select("country_norm", "tok", ((pl.col("N") + 1) / (pl.col("len") + 1)).log().cast(pl.Float32).alias("idf")))
    return side(s1), side(pool), idf


def _idf_agg(lst: pl.Series, country: pl.Series, idf: pl.DataFrame, default: float, name: str) -> pl.DataFrame:
    x = (pl.DataFrame({"r": np.arange(len(lst), dtype=np.uint32), "tok": lst, "country_norm": country})
         .explode("tok").drop_nulls("tok").join(idf, on=["country_norm", "tok"], how="left")
         .with_columns(pl.col("idf").fill_null(default))
         .group_by("r").agg(pl.col("idf").sum().alias(f"{name}_idf_sum"), pl.col("idf").max().alias(f"{name}_idf_max")))
    return (pl.DataFrame({"r": np.arange(len(lst), dtype=np.uint32)}).join(x, on="r", how="left")
            .sort("r").drop("r").fill_null(0).cast(pl.Float32))


def name_evidence_features(pairs: pl.DataFrame, s1n: pl.DataFrame, pooln: pl.DataFrame, idf: pl.DataFrame,
                           unseen_idf: float = math.log(1e6)) -> pl.DataFrame:
    A = s1n.select(pl.all().gather(pairs["s1_idx"]))
    B = pooln.select(pl.all().gather(pairs["pool_idx"]))
    ta, tb = A["name_tok"], B["name_tok"]
    d = pl.DataFrame({"a": ta, "b": tb})
    sets = d.select(pl.col("a").list.set_intersection(pl.col("b")).alias("sh"),
                    pl.col("a").list.set_difference(pl.col("b")).alias("ao"),
                    pl.col("b").list.set_difference(pl.col("a")).alias("bo"))
    country = A["country_norm"]
    sh = _idf_agg(sets["sh"], country, idf, unseen_idf, "shared")
    ao = _idf_agg(sets["ao"], country, idf, unseen_idf, "s1only")
    bo = _idf_agg(sets["bo"], country, idf, unseen_idf, "poolonly")
    tot = sh["shared_idf_sum"] + ao["s1only_idf_sum"] + bo["poolonly_idf_sum"]
    left_a = sets["ao"].list.join(" ").to_list()
    left_b = sets["bo"].list.join(" ").to_list()
    lr = process.cpdist(left_a, left_b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    both_empty = (sets["ao"].list.len() == 0) & (sets["bo"].list.len() == 0)
    return pl.concat([
        sh, ao, bo,
        pl.DataFrame({
            "idf_jacc": (sh["shared_idf_sum"] / tot.clip(1e-6)).cast(pl.Float32),
            "leftover_ratio": pl.Series(lr).set(both_empty, 100.0),
            "n_s1_same_name_a": A["n_s1_same_name"], "n_pool_same_name_a": A["n_pool_same_name"],
            "n_s1_same_name_b": B["n_s1_same_name"], "n_pool_same_name_b": B["n_pool_same_name"],
        }),
    ], how="horizontal")
