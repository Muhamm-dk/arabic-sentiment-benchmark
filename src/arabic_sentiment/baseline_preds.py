"""Refit the validation-selected TF-IDF config per dataset and save test predictions.
Run: uv run python -m arabic_sentiment.baseline_preds
Output: results/preds/<dataset>-tfidf-s0.parquet, and a check against results/baseline.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from arabic_sentiment.baseline import normalize, vectorizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    for b in json.load(open(ROOT / "results/baseline.json")):
        d, c = b["dataset"], b["config"]
        df = pd.read_parquet(ROOT / f"data/processed/{d}.parquet")
        tr, te = df[df.split == "train"], df[df.split == "test"].copy()
        f = (lambda s: s.map(normalize)) if c["normalize"] else (lambda s: s)
        v = vectorizer(c["features"])
        X = v.fit_transform(f(tr.text))
        m = LogisticRegression(C=c["C"], solver="liblinear", max_iter=1000).fit(X, tr.label)
        te["pred"] = m.predict(v.transform(f(te.text)))
        score = f1_score(te.label, te.pred, average="macro")
        print(f"{d}: refit {score:.4f} vs saved {b['test_macro_f1']:.4f}")
        te.to_parquet(ROOT / f"results/preds/{d}-tfidf-s0.parquet", index=False)


if __name__ == "__main__":
    main()
