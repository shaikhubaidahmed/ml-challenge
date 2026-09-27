"""Loading raw TSVs into cached, normalized parquet tables.

Per split we build two tables:
  s1   : Source 1 records, idx = 0..N1-1
  pool : Source 2 + Source 3 records, idx = 0..N23-1, src in {2,3}
For the train split, pool.gid holds the s1.idx of the matching S1 entity (-1 if none);
the ground truth guarantees each S2/S3 record matches at most one S1 entity (audit, fact H).
"""
from pathlib import Path

import polars as pl

from . import translit
from .normalize import DEVA, add_normalized, raw_tokens

RENAME = {"entity_id": "eid", "business_name": "name", "business_address": "addr"}


def read_tsv(path) -> pl.DataFrame:
    """Read a challenge TSV with quoting disabled and every column as string."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def build_split(cfg, split: str, force=False):
    work = Path(cfg["paths"]["work_dir"])
    s1_path, pool_path = work / f"{split}_s1.parquet", work / f"{split}_pool.parquet"
    if s1_path.exists() and pool_path.exists() and not force:
        return pl.read_parquet(s1_path), pl.read_parquet(pool_path)
    d = Path(cfg["paths"]["data_dir"]) / split
    s1 = read_tsv(d / f"{split}_source1.tsv").rename(RENAME)
    s1 = s1.with_row_index("idx").with_columns(pl.lit(1, pl.UInt8).alias("src"))
    parts = [read_tsv(d / f"{split}_source{k}.tsv").rename(RENAME).with_columns(pl.lit(k, pl.UInt8).alias("src"))
             for k in (2, 3)]
    pool = pl.concat(parts).with_row_index("idx")
    if split == "train":
        gt = read_tsv(d / "train_ground_truth.tsv")
        links = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
                 .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
                 .join(s1.select(pl.col("eid").alias("source1_entity_id"), pl.col("idx").alias("gid")),
                       on="source1_entity_id")
                 .select(pl.col("matched_entity_ids").alias("eid"), "gid"))
        pool = pool.join(links, on="eid", how="left").with_columns(pl.col("gid").fill_null(-1).cast(pl.Int64))
        pool = pool.sort("idx")
    s1, pool = raw_tokens(s1), raw_tokens(pool)
    tmap = None
    if cfg.get("translit", False):
        tpath = work / "translit.json"
        if split == "train":
            deva = pool.with_columns(pl.col("name").str.contains(DEVA).alias("name_deva"))
            deva = deva.filter(pl.col("name").str.contains(DEVA) | pl.col("addr").str.contains(DEVA))
            tmap = translit.learn(s1, deva)
            translit.save(tmap, tpath)
        else:
            if not tpath.exists():
                raise FileNotFoundError(f"{tpath} missing: build the train split first (map is learned on train only)")
            tmap = translit.load(tpath)
    s1, pool = add_normalized(s1, tmap), add_normalized(pool, tmap)
    s1.write_parquet(s1_path)
    pool.write_parquet(pool_path)
    return s1, pool
