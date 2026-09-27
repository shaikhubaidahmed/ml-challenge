"""Matching models (Phase C) and decision rule (Phase D)."""
import numpy as np
import polars as pl

from .config import preload_libomp

preload_libomp()
import lightgbm as lgb  # noqa: E402

NON_FEATURES = {"s1_idx", "pool_idx", "y", "fold"}

STAGE1_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     max_bin=255, num_threads=0, verbose=-1, seed=42)
STAGE2_PARAMS = dict(STAGE1_PARAMS, num_leaves=127, learning_rate=0.05)


def feature_cols(df: pl.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURES]


def train_lgb(train: pl.DataFrame, valid: pl.DataFrame, cols, params, rounds=600, early=50):
    dtr = lgb.Dataset(train.select(cols).to_numpy(), label=train["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(valid.select(cols).to_numpy(), label=valid["y"].to_numpy(), reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(100)])


def predict(model, df: pl.DataFrame, cols) -> np.ndarray:
    return model.predict(df.select(cols).to_numpy(), num_threads=0).astype(np.float32)


def context_features(df: pl.DataFrame, p="p1") -> pl.DataFrame:
    """Stage-2 context from stage-1 probabilities: within-S1 and within-pool-record competition."""
    s = pl.col(p)
    df = df.with_columns(
        s.rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32).alias(f"{p}_rank_s1"),
        (s.max().over("s1_idx") - s).cast(pl.Float32).alias(f"{p}_gap_s1max"),
        s.rank("ordinal", descending=True).over("pool_idx").cast(pl.Float32).alias(f"{p}_rank_pool"),
        (s.max().over("pool_idx") - s).cast(pl.Float32).alias(f"{p}_gap_poolmax"),
        s.sum().over("pool_idx").cast(pl.Float32).alias(f"{p}_sum_pool"),
        s.sum().over("s1_idx").cast(pl.Float32).alias(f"{p}_sum_s1"),
        (s > 0.5).sum().over("s1_idx").cast(pl.Float32).alias(f"{p}_n05_s1"),
        (s > 0.5).sum().over("pool_idx").cast(pl.Float32).alias(f"{p}_n05_pool"),
    )
    # best competing S1 for the same pool record (excluding this pair)
    top2 = s.top_k(2).over("pool_idx", mapping_strategy="join")
    return df.with_columns(
        pl.when(s >= top2.list.first()).then(top2.list.get(1, null_on_oob=True)).otherwise(top2.list.first())
        .fill_null(0).cast(pl.Float32).alias(f"{p}_best_other_s1")
    )


def decide(scores: pl.DataFrame, thr: float, p="p2", exclusive=True) -> pl.DataFrame:
    """Select matches: score >= thr, and (exclusive) each pool record goes to at most its best S1."""
    d = scores.filter(pl.col(p) >= thr)
    if exclusive:
        d = d.filter(pl.col(p) == pl.col(p).max().over("pool_idx")).unique("pool_idx", keep="first")
    return d.select("s1_idx", "pool_idx")
