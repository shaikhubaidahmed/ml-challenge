"""Per-entity expected-F0.5 decision rule (Phase D alternative to a global threshold).

For one S1 entity with candidate probabilities p_1 >= p_2 >= ... (after exclusive assignment),
predicting the top-k set has approximate expected F_beta
    E[F](k) ~= (1+b2) * sum_{i<=k} p_i / (k + b2 * N),   N = expected number of true matches
with N = sum_i p_i + extra (extra accounts for true matches outside the candidate list).
Predicting nothing scores 1 only if the entity is a singleton: E[F](0) = prod_i (1 - p_i) * (1 - miss).
The k with the highest value is chosen. Probabilities should be calibrated (OOF-trained models are).
"""
import polars as pl


def expected_f_select(scores: pl.DataFrame, p="p2", beta=0.5, extra=0.0, min_p=0.05, empty_scale=1.0) -> pl.DataFrame:
    b2 = beta * beta
    d = scores.filter(pl.col(p) >= min_p).sort(["s1_idx", p], descending=[False, True])
    d = d.with_columns(
        pl.col(p).cum_sum().over("s1_idx").alias("_tp"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("_k"),
        (pl.col(p).sum().over("s1_idx") + extra).alias("_n"),
    )
    d = d.with_columns(((1 + b2) * pl.col("_tp") / (pl.col("_k") + b2 * pl.col("_n"))).alias("_ef"))
    # expected score of predicting nothing (computed on all candidates, not only >= min_p)
    empty = scores.group_by("s1_idx").agg(((1 - pl.col(p)).clip(1e-9, 1).log().sum().exp() * empty_scale).alias("_e0"))
    best = d.group_by("s1_idx").agg(pl.col("_ef").max().alias("_best"), pl.col("_k").get(pl.col("_ef").arg_max()).alias("_kbest"))
    best = best.join(empty, on="s1_idx", how="left").filter(pl.col("_best") > pl.col("_e0").fill_null(0))
    return (d.join(best.select("s1_idx", "_kbest"), on="s1_idx")
            .filter(pl.col("_k") <= pl.col("_kbest")).select("s1_idx", "pool_idx"))
