"""Checks on the deployed (combined) model. Run: uv run python -m arabic_sentiment.combined_checks
(1) paired bootstrap against the single-dataset MARBERT runs on identical test rows
(2) calibration of its confidence per dataset
"""
import numpy as np
import pandas as pd

from arabic_sentiment.error_extra import BINS
from arabic_sentiment.paired_all import N_BOOT, PREDS, SEED, find, macro_f1

DATASETS = ["labr", "hard_notitle", "astd", "ajgt"]


def paired(comb):
    print("== combined minus single-dataset MARBERT (points, paired 95% interval for seed 0)")
    for d in DATASETS:
        A = comb[comb.dataset == d][["text", "label", "pred"]]
        res = {}
        for s in (0, 1, 2):
            f = find(d, "marbert", s)
            if f is None:
                continue
            B = pd.read_parquet(f)[["text", "pred"]]
            m = A.merge(B, on="text", suffixes=("_a", "_b"))
            assert len(m) == len(A) == len(B), f"{d} seed {s}: matched {len(m)} of {len(A)} and {len(B)}"
            y, pa, pb = m.label.to_numpy(), m.pred_a.to_numpy(), m.pred_b.to_numpy()
            res[s] = (macro_f1(y, pa) - macro_f1(y, pb), y, pa, pb)
        diff0, y, pa, pb = res[0]
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT)
        for k in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            boots[k] = macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        others = ", ".join(f"{v[0] * 100:+.1f}" for s, v in res.items() if s)
        print(f"{d:13s} n={len(y):5d} | seed 0: {diff0 * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
              + (f" | seeds 1, 2: {others}" if others else ""))


def calibration(comb):
    for d in DATASETS:
        g = comb[comb.dataset == d]
        conf = np.maximum(g.prob_pos, 1 - g.prob_pos).to_numpy()
        ok = (g.pred == g.label).to_numpy()
        t = pd.DataFrame({"bin": pd.cut(conf, BINS, right=False), "conf": conf, "ok": ok})
        t = t.groupby("bin", observed=True).agg(n=("ok", "size"), mean_conf=("conf", "mean"), accuracy=("ok", "mean"))
        ece = float((t.n / t.n.sum() * (t.accuracy - t.mean_conf).abs()).sum())
        hi = conf >= 0.9
        print(f"\n== combined model calibration, {d} (n={len(g)}), expected calibration error {ece:.3f}")
        print(t.round(3).to_string())
        print(f"confidence 0.9 or more: {int(hi.sum())} rows, accuracy {ok[hi].mean():.3f}, mean confidence {conf[hi].mean():.3f}")


def main():
    comb = pd.read_parquet(PREDS / "combined-marbert-s0.parquet")
    paired(comb)
    calibration(comb)


if __name__ == "__main__":
    main()

