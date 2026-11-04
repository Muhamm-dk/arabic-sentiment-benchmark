"""Quantitative error statistics for Chapter 14 (seed 0 predictions, test sets).
Run: uv run python -m arabic_sentiment.error_stats
"""
import re

import numpy as np
import pandas as pd

from arabic_sentiment.dialect_analysis import ROOT, tag_frame

SYSTEMS = ["tfidf", "arabert", "marbert", "xlmr"]
DATASETS = ["ajgt", "astd", "labr", "hard_notitle"]


def find(d, m):
    pat = re.compile(rf"{d}-{m}-s0(-n\d+)?\.parquet")
    return next(f for f in (ROOT / "results/preds").glob(f"{d}-{m}-s0*.parquet") if pat.fullmatch(f.name))


def load(d):
    df = pd.read_parquet(find(d, "marbert"))[["text", "label", "pred", "prob_pos"]].rename(
        columns={"pred": "pred_marbert"})
    n = len(df)
    for m in SYSTEMS:
        if m == "marbert":
            continue
        x = pd.read_parquet(find(d, m))[["text", "pred"]].rename(columns={"pred": f"pred_{m}"})
        df = df.merge(x, on="text")
        assert len(df) == n, f"{d}: row mismatch after merging {m}"
    return df


def main():
    tags = pd.read_parquet(ROOT / "results/dialect_tags.parquet")
    for d in DATASETS:
        df = load(d)
        W = pd.DataFrame({m: df[f"pred_{m}"] != df.label for m in SYSTEMS})
        me = W["marbert"]
        df["err"] = me
        conf = np.maximum(df.prob_pos, 1 - df.prob_pos)
        print(f"\n== {d} (n={len(df)})")
        print("error rate:", {m: round(float(W[m].mean()), 3) for m in SYSTEMS})
        fp = int(((df.label == 0) & (df.pred_marbert == 1)).sum())
        fn = int(((df.label == 1) & (df.pred_marbert == 0)).sum())
        print(f"MARBERT errors {int(me.sum())}: {fp} false positives (gold negative, predicted positive), {fn} false negatives")
        others = [m for m in SYSTEMS if m != "marbert"]
        shared = me & (W[others].sum(axis=1) >= 2)
        print(f"wrong for all four systems: {int(W.all(axis=1).sum())}"
              f" | MARBERT errors that at least two other systems also make: {int(shared.sum())} of {int(me.sum())}")
        print(f"MARBERT confidence: mean {conf[~me].mean():.3f} when right, {conf[me].mean():.3f} when wrong;"
              f" errors at confidence 0.9 or more: {int((conf[me] >= 0.9).sum())} of {int(me.sum())}")
        df["q"] = pd.qcut(df.text.str.split().str.len(), 4, duplicates="drop")
        print("error rate by length quartile (words):")
        print(df.groupby("q", observed=True).err.agg(["size", "mean"]).round(3).to_string())
        g = df.merge(tag_frame(tags, d)[["text", "dialect"]], on="text", how="left")
        assert len(g) == len(df)
        print("error rate by predicted dialect (any confidence):")
        print(g.groupby("dialect").err.agg(["size", "mean"]).round(3).to_string())
        allw = W.all(axis=1)
        top = df[allw].assign(conf=conf[allw]).sort_values("conf", ascending=False).head(8)
        print("wrong for all four systems, highest MARBERT confidence:")
        for _, r in top.iterrows():
            print(f"  gold={int(r.label)} conf={r.conf:.2f} | {r.text[:110]}")


if __name__ == "__main__":
    main()
