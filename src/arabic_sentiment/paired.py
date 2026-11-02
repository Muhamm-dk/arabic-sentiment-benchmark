"""Paired bootstrap of the macro-F1 difference between two prediction files on identical rows.
Run: uv run python -m arabic_sentiment.paired results/preds/A.parquet results/preds/B.parquet
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)


def main(a_path, b_path, n=2000, seed=42):
    a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
    m = a[["text", "label", "pred"]].merge(b[["text", "pred"]], on="text", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), f"row mismatch: {len(a)} and {len(b)} rows, {len(m)} merged"
    y, pa, pb = m.label.to_numpy(), m.pred_a.to_numpy(), m.pred_b.to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1(y[i], pa[i]) - f1(y[i], pb[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"A {f1(y, pa):.4f} | B {f1(y, pb):.4f} | A-B {f1(y, pa) - f1(y, pb):+.4f}"
          f" | 95% CI [{lo:+.4f}, {hi:+.4f}] | n={len(y)}")
    print("A right and B wrong:", int(((pa == y) & (pb != y)).sum()),
          "| A wrong and B right:", int(((pa != y) & (pb == y)).sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
