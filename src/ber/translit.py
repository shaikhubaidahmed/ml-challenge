"""Indic-script (Devanagari, Bengali, Gujarati, Tamil, ...) -> Latin token dictionary learned ONLY from training ground-truth pairs.

Names: Devanagari names have the same token count as their S1 name in 99.997% of train
positives, so tokens are aligned by position. Addresses: token counts differ, so each
Devanagari address token is mapped to the S1 address token with the highest co-occurrence
lift P(latin | deva) / P(latin).
"""
import json

import polars as pl

DEVA = r"[\u0900-\u0dff]"  # any Indic script block
_DEVANAGARI = r"[\u0900-\u097f]"


def learn(s1: pl.DataFrame, pool: pl.DataFrame, min_count=3, min_conf=0.5) -> dict:
    pos = pool.filter((pl.col("gid") >= 0)).select("gid", "name_tok", "addr_tok", "name_deva")
    s1 = s1.select(pl.col("idx").cast(pl.Int64).alias("gid"), pl.col("name_tok").alias("s1_name"),
                   pl.col("addr_tok").alias("s1_addr"))
    d = pos.join(s1, on="gid")

    # names: positional alignment
    nm = (d.filter(pl.col("name_deva") & (pl.col("name_tok").list.len() == pl.col("s1_name").list.len()))
          .select(pl.col("name_tok").alias("d"), pl.col("s1_name").alias("l")).explode(["d", "l"])
          .filter(pl.col("d").str.contains(DEVA)))
    cnt = nm.group_by("d", "l").len()
    best = (cnt.sort("len", descending=True).group_by("d")
            .agg(pl.col("l").first(), pl.col("len").first().alias("top"), pl.col("len").sum().alias("tot"))
            .filter((pl.col("tot") >= min_count) & (pl.col("top") / pl.col("tot") >= min_conf)))
    mapping = dict(zip(best["d"].to_list(), best["l"].to_list()))

    # addresses: co-occurrence lift
    ad = d.filter(pl.col("addr_tok").list.eval(pl.element().str.contains(DEVA)).list.any())
    pairs = (ad.select(pl.int_range(pl.len()).alias("r"), pl.col("addr_tok").alias("d"), pl.col("s1_addr").alias("l"))
             .explode("d").filter(pl.col("d").str.contains(DEVA)).explode("l").unique())
    n_rows = ad.height
    lf = (ad.select(pl.col("s1_addr").list.unique().alias("l")).explode("l").group_by("l").len()
          .select("l", (pl.col("len") / n_rows).alias("pl")))
    dc = pairs.select("r", "d").unique().group_by("d").len().rename({"len": "nd"})
    cooc = (pairs.group_by("d", "l").len().join(dc, on="d").join(lf, on="l")
            .with_columns((pl.col("len") / pl.col("nd")).alias("cond"))
            .with_columns((pl.col("cond") / pl.col("pl")).alias("lift"))
            .filter((pl.col("nd") >= min_count) & (pl.col("cond") >= 0.8)))
    abest = cooc.sort("lift", descending=True).group_by("d").agg(pl.col("l").first(), pl.col("cond").first())
    for dtok, ltok in zip(abest["d"].to_list(), abest["l"].to_list()):
        mapping.setdefault(dtok, ltok)
    return mapping


def save(mapping: dict, path):
    with open(path, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=0, sort_keys=True)


def load(path) -> dict:
    with open(path) as f:
        return json.load(f)
