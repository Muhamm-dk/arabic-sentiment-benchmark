"""More error statistics for Chapter 14. Run: uv run python -m arabic_sentiment.error_extra
(1) error rate by original star rating (LABR and HARD without the descriptor), all four systems
(2) calibration of MARBERT's confidence
"""
import re

import numpy as np
import pandas as pd

from arabic_sentiment.dialect_analysis import ROOT

SYSTEMS = ["tfidf", "arabert", "marbert", "xlmr"]
BINS = [0.5, 0.7, 0.9, 0.99, 1.0001]


def find(d, m):
    pat = re.compile(rf"{d}-{m}-s0(-n\d+)?\.parquet")
    return next(f for f in (ROOT / "results/preds").glob(f"{d}-{m}-s0*.parquet") if pat.fullmatch(f.name))


def stars_table():
    for d in ["labr", "hard_notitle"]:
        base = pd.read_parquet(find(d, "marbert"))[["text", "label", "stars"]]
        out = base.groupby("stars").size().rename("n").to_frame()
        for m in SYSTEMS:
            p = pd.read_parquet(find(d, m))[["text", "pred"]]
            x = base.merge(p, on="text")
            assert len(x) == len(base), f"{d} {m}: row mismatch"
            out[m] = (x.pred != x.label).groupby(x.stars).mean().round(3)
        print(f"\n== error rate by original star rating, {d}")
        print(out.to_string())


def calibration():
    for d in ["ajgt", "astd", "labr", "hard_notitle"]:
        p = pd.read_parquet(find(d, "marbert"))
        conf = np.maximum(p.prob_pos, 1 - p.prob_pos).to_numpy()
        ok = (p.pred == p.label).to_numpy()
        t = pd.DataFrame({"bin": pd.cut(conf, BINS, right=False), "conf": conf, "ok": ok})
        t = t.groupby("bin", observed=True).agg(n=("ok", "size"), mean_conf=("conf", "mean"), accuracy=("ok", "mean"))
        ece = float((t.n / t.n.sum() * (t.accuracy - t.mean_conf).abs()).sum())
        hi = conf >= 0.9
        print(f"\n== MARBERT calibration, {d} (n={len(p)}), expected calibration error {ece:.3f}")
        print(t.round(3).to_string())
        print(f"confidence 0.9 or more: {int(hi.sum())} rows, accuracy {ok[hi].mean():.3f}, mean confidence {conf[hi].mean():.3f}")


if __name__ == "__main__":
    stars_table()
    calibration()
