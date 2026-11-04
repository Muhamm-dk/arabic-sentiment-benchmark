"""Is the 300-row LABR sample unusually easy? Compare its score shifts with random samples of the same shape.
Run: uv run python -m arabic_sentiment.sample_check
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ["tfidf", "arabert", "marbert", "xlmr"]
N_DRAWS, SEED = 10000, 42


def macro_f1(y, p):
    tp1, tp0 = np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 0))
    fp1, fn1 = np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p == 0))
    f1_pos = 2 * tp1 / max(2 * tp1 + fp1 + fn1, 1)
    f1_neg = 2 * tp0 / max(2 * tp0 + fp1 + fn1, 1)
    return (f1_pos + f1_neg) / 2


def main():
    path = ROOT / "results/llm/qwen-labr-test-en.jsonl"
    sample = {json.loads(line)["text"] for line in open(path, encoding="utf-8")}
    frames = {m: pd.read_parquet(ROOT / f"results/preds/labr-{m}-s0.parquet")[["text", "label", "pred"]]
              for m in SYSTEMS}
    texts = frames["tfidf"].text
    assert texts.is_unique
    y = frames["tfidf"].label.to_numpy()
    preds = {}
    for m, f in frames.items():
        f = f.set_index("text").loc[texts]
        assert (f.label.to_numpy() == y).all(), f"{m}: labels differ"
        preds[m] = f.pred.to_numpy()
    in_s = texts.isin(sample).to_numpy()
    assert in_s.sum() == len(sample), f"matched {in_s.sum()} of {len(sample)} sample rows"
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos, n_neg = int(y[in_s].sum()), int((y[in_s] == 0).sum())
    full = {m: macro_f1(y, preds[m]) for m in SYSTEMS}
    obs = {m: macro_f1(y[in_s], preds[m][in_s]) - full[m] for m in SYSTEMS}
    rng = np.random.default_rng(SEED)
    shifts = {m: np.empty(N_DRAWS) for m in SYSTEMS}
    for k in range(N_DRAWS):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
        for m in SYSTEMS:
            shifts[m][k] = macro_f1(y[idx], preds[m][idx]) - full[m]
    print(f"sample: {in_s.sum()} rows ({n_pos} positive, {n_neg} negative), {N_DRAWS} random draws of that shape")
    for m in SYSTEMS:
        s = shifts[m]
        print(f"{m:8s} full {full[m]:.4f} | observed shift {obs[m] * 100:+.1f} | simulated mean {s.mean() * 100:+.2f},"
              f" sd {s.std() * 100:.2f} | share of draws at least as high {np.mean(s >= obs[m]):.4f}")
    both = np.mean((shifts["tfidf"] >= obs["tfidf"]) & (shifts["marbert"] >= obs["marbert"]))
    every = np.mean(np.all([shifts[m] >= obs[m] for m in SYSTEMS], axis=0))
    print("TF-IDF and MARBERT both at least as high:", round(float(both), 4))
    print("all four at least as high:", round(float(every), 4))


if __name__ == "__main__":
    main()
