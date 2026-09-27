"""Phase F: write submission TSVs (one row per Source 1 entity, comma-joined ids, no quoting)."""
import polars as pl


def write_id_lists(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, path, col: str):
    """pairs: (s1_idx, pool_idx). Every S1 entity gets exactly one row; empty list when no pairs."""
    ids = (pairs.unique(["s1_idx", "pool_idx"])
           .join(pool.select(pl.col("idx").cast(pl.UInt32).alias("pool_idx"), pl.col("eid").alias("oid")),
                 on="pool_idx")
           .sort("s1_idx", "oid")
           .group_by("s1_idx", maintain_order=True).agg(pl.col("oid").str.join(",").alias(col)))
    out = (s1.select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), pl.col("eid").alias("source1_entity_id"))
           .join(ids.with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx", how="left")
           .select("source1_entity_id", pl.col(col).fill_null("")))
    assert out.height == s1.height and out["source1_entity_id"].n_unique() == s1.height
    out.write_csv(path, separator="\t", quote_style="never")
