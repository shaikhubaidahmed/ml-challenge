"""Official metric (macro F0.5 per Source 1 entity) and blocking diagnostics."""
import numpy as np
import polars as pl


def macro_f05(pred: pl.DataFrame, s1_idx: np.ndarray, pool: pl.DataFrame, beta=0.5) -> dict:
    """pred: (s1_idx, pool_idx) predicted matches. Truth from pool.gid. Evaluated over s1_idx entities."""
    ev = pl.DataFrame({"s1_idx": s1_idx.astype(np.int64)})
    truth = pool.filter(pl.col("gid") >= 0).group_by(pl.col("gid").alias("s1_idx")).agg(pl.len().alias("n_true"))
    p = pred.select(pl.col("s1_idx").cast(pl.Int64), pl.col("pool_idx").cast(pl.UInt32))
    p = p.join(pool.select(pl.col("idx").cast(pl.UInt32).alias("pool_idx"), "gid"), on="pool_idx", how="left")
    per = p.group_by("s1_idx").agg(pl.len().alias("n_pred"), (pl.col("gid") == pl.col("s1_idx")).sum().alias("tp"))
    d = ev.join(truth, on="s1_idx", how="left").join(per, on="s1_idx", how="left").fill_null(0)
    tp, npred, ntrue = (d[c].to_numpy().astype(float) for c in ("tp", "n_pred", "n_true"))
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(prec + rec > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    both_empty = (npred == 0) & (ntrue == 0)
    f = np.where(both_empty, 1.0, f)
    single = ntrue == 0
    return {
        "f05": float(f.mean()),
        "precision_macro": float(np.where(npred > 0, prec, 1.0 * (ntrue == 0)).mean()),
        "recall_macro": float(np.where(ntrue > 0, rec, 1.0 * (npred == 0)).mean()),
        "singleton_acc": float(f[single].mean()) if single.any() else float("nan"),
        "nonsingleton_f05": float(f[~single].mean()),
        "pair_precision": float(tp.sum() / max(npred.sum(), 1)),
        "pair_recall": float(tp.sum() / max(ntrue.sum(), 1)),
        "n_entities": int(len(f)),
    }


def blocking_report(cands: pl.DataFrame, s1_idx: np.ndarray, pool: pl.DataFrame) -> dict:
    """Candidate recall ceiling and size statistics for the given S1 entities."""
    ev = pl.DataFrame({"s1_idx": s1_idx.astype(np.int64)})
    c = cands.select(pl.col("s1_idx").cast(pl.Int64), pl.col("pool_idx").cast(pl.UInt32)).join(ev, on="s1_idx")
    c = c.join(pool.select(pl.col("idx").cast(pl.UInt32).alias("pool_idx"), "gid"), on="pool_idx", how="left")
    tp = int((c["gid"] == c["s1_idx"]).sum())
    truth = pool.filter(pl.col("gid").is_in(ev["s1_idx"].implode()))
    counts = c.group_by("s1_idx").len()["len"].to_numpy()
    n_tot = truth.height
    # entity-level ceiling: F0.5 if the model were perfect on the candidate set
    per = c.group_by("s1_idx").agg((pl.col("gid") == pl.col("s1_idx")).sum().alias("tp"))
    tr = truth.group_by(pl.col("gid").alias("s1_idx")).agg(pl.len().alias("n"))
    d = ev.join(tr, on="s1_idx", how="left").join(per, on="s1_idx", how="left").fill_null(0)
    tpn, nn = d["tp"].to_numpy().astype(float), d["n"].to_numpy().astype(float)
    rec = np.where(nn > 0, tpn / np.maximum(nn, 1), 1.0)
    f_ceiling = np.where(nn > 0, np.where(tpn > 0, 1.25 * rec / (0.25 + rec), 0.0), 1.0)
    return {
        "pair_recall": tp / max(n_tot, 1),
        "true_pairs": n_tot,
        "cand_pairs": c.height,
        "mean_cands": float(counts.mean()) if len(counts) else 0.0,
        "median_cands": float(np.median(counts)) if len(counts) else 0.0,
        "max_cands": int(counts.max()) if len(counts) else 0,
        "s1_zero_cands": int(len(s1_idx) - len(counts)),
        "f05_ceiling": float(f_ceiling.mean()),
    }


def s1_subset_mask(s1_idx: pl.Expr, frac: float) -> pl.Expr:
    """Deterministic S1-entity subset used consistently by feature building and evaluation."""
    return (s1_idx.cast(pl.UInt64).hash(seed=5) % 1000) < int(round(frac * 1000))
