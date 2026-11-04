"""Checks for Chapter 13. Run: uv run python -m arabic_sentiment.dialect_checks
(1) Do the two dialect classifiers agree?  (2) Is the MSA vs other-dialect score gap larger than noise?
"""
import numpy as np
import pandas as pd

from arabic_sentiment.dialect_analysis import ENCODERS, MIN_CONF, ROOT, encoder_preds, tag_frame

N_BOOT, SEED = 2000, 42
MAP = {"GLF": "Gulf", "EGY": "Egyptian", "LEV": "Levantine", "MAGHREB": "Maghrebi"}


def macro_f1(y, p):
    tp1, tp0 = np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 0))
    fp1, fn1 = np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p == 0))
    return (2 * tp1 / max(2 * tp1 + fp1 + fn1, 1) + 2 * tp0 / max(2 * tp0 + fp1 + fn1, 1)) / 2


def agreement(tags):
    print(f"== agreement of the two classifiers (rows with main-classifier confidence >= {MIN_CONF})")
    for d, g in tags.groupby("dataset"):
        g = g[g.conf >= MIN_CONF]
        non, msa = g[g.dialect != "MSA"], g[g.dialect == "MSA"]
        agree = (non.dialect.map(MAP) == non.dialect_qadi).mean()
        print(f"{d}: non-MSA {len(non)} rows, second classifier agrees on {agree:.1%}; MSA {len(msa)} rows, "
              f"second classifier says {msa.dialect_qadi.value_counts(normalize=True).round(2).to_dict()}")


def group_diff(tags):
    print("\n== score difference between groups (first minus second), each group resampled independently")
    rng = np.random.default_rng(SEED)
    for d in ["labr", "hard", "hard_notitle", "astd"]:
        t = tag_frame(tags, d)
        t = t[t.conf >= MIN_CONF]
        for m in ENCODERS:
            p = encoder_preds(d, m)
            if p is None:
                continue
            x = p.merge(t[["text", "dialect"]], on="text")
            a = x[x.dialect == "MSA"]
            b = x[x.dialect == "EGY"] if d == "astd" else x[x.dialect != "MSA"]
            ya, pa, yb, pb = a.label.to_numpy(), a.pred.to_numpy(), b.label.to_numpy(), b.pred.to_numpy()
            diffs = np.empty(N_BOOT)
            for k in range(N_BOOT):
                i, j = rng.integers(0, len(ya), len(ya)), rng.integers(0, len(yb), len(yb))
                diffs[k] = macro_f1(ya[i], pa[i]) - macro_f1(yb[j], pb[j])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            d0 = macro_f1(ya, pa) - macro_f1(yb, pb)
            name = "MSA - EGY" if d == "astd" else "MSA - non-MSA"
            print(f"{d:13s} {m:8s} {name:14s} {d0 * 100:+.1f} points [{lo * 100:+.1f}, {hi * 100:+.1f}]"
                  f" | n {len(ya)} and {len(yb)}")


def main():
    tags = pd.read_parquet(ROOT / "results/dialect_tags.parquet")
    agreement(tags)
    group_diff(tags)


if __name__ == "__main__":
    main()
