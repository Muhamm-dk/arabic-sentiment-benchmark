"""Paired bootstrap of macro-F1 differences between systems on identical test rows.
Run: uv run python -m arabic_sentiment.paired_all
Output: results/paired.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "results/preds"
DATASETS = ["labr", "hard", "hard_notitle", "astd", "ajgt"]
PAIRS = [("arabert", "tfidf"), ("marbert", "tfidf"), ("xlmr", "tfidf"),
         ("marbert", "arabert"), ("marbert", "xlmr"), ("arabert", "xlmr")]
N_BOOT, SEED = 2000, 42


def macro_f1(y, p):
    tp1, tp0 = np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 0))
    fp1, fn1 = np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p == 0))
    return (2 * tp1 / max(2 * tp1 + fp1 + fn1, 1) + 2 * tp0 / max(2 * tp0 + fp1 + fn1, 1)) / 2


def find(d, m, s):
    pat = re.compile(rf"{d}-{m}-s{s}(-n\d+)?\.parquet")
    hits = [f for f in PREDS.glob(f"{d}-{m}-s{s}*.parquet") if pat.fullmatch(f.name)]
    return hits[0] if hits else None


def main():
    rows = []
    for d in DATASETS:
        for a, b in PAIRS:
            res = {}
            for s in (0, 1, 2):
                fa, fb = find(d, a, s), find(d, b, 0 if b == "tfidf" else s)
                if fa is None or fb is None:
                    continue
                A = pd.read_parquet(fa)[["text", "label", "pred"]]
                B = pd.read_parquet(fb)[["text", "pred"]]
                m = A.merge(B, on="text", suffixes=("_a", "_b"))
                assert len(m) == len(A) == len(B), f"{d} {a} vs {b} seed {s}: row mismatch"
                y, pa, pb = m.label.to_numpy(), m.pred_a.to_numpy(), m.pred_b.to_numpy()
                res[s] = (macro_f1(y, pa) - macro_f1(y, pb), y, pa, pb)
            if 0 not in res:
                continue
            diff0, y, pa, pb = res[0]
            rng = np.random.default_rng(SEED)
            boots = np.empty(N_BOOT)
            for k in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                boots[k] = macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"dataset": d, "pair": f"{a} - {b}", "n": len(y), "diff_s0": round(diff0, 4),
                         "ci95": f"[{lo:+.4f}, {hi:+.4f}]", "excludes_0": bool(lo > 0 or hi < 0),
                         "diff_s1": round(res[1][0], 4) if 1 in res else None,
                         "diff_s2": round(res[2][0], 4) if 2 in res else None})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/paired.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
