"""Does stripping only unquoted descriptors change the HARD no-descriptor TF-IDF score?
Run: uv run python -m arabic_sentiment.hard_strict_check
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from arabic_sentiment.baseline import normalize, vectorizer
from arabic_sentiment.data import DESCRIPTORS, lead, strip_descriptor

ROOT = Path(__file__).resolve().parents[2]
QUOTE_STARTS = ("\u201c", '"')


def strict_strip(text):
    if text.lstrip().startswith(QUOTE_STARTS):
        return text
    return strip_descriptor(text)


def predict(train, test, col, cfg):
    f = normalize if cfg["normalize"] else (lambda s: s)
    v = vectorizer(cfg["features"])
    X = v.fit_transform(train[col].map(f))
    m = LogisticRegression(C=cfg["C"], solver="liblinear", max_iter=1000).fit(X, train.label)
    return m.predict(v.transform(test[col].map(f)))


def main():
    base = json.load(open(ROOT / "results/baseline.json"))
    cfg = next(b["config"] for b in base if b["dataset"] == "hard_notitle")
    d = pd.read_parquet(ROOT / "data/processed/hard.parquet")
    d["current"] = d.text.map(strip_descriptor)
    d["strict"] = d.text.map(strict_strip)
    quoted = d.text.str.lstrip().str.startswith(QUOTE_STARTS)
    flagged = d.text.map(lead).isin(DESCRIPTORS) & quoted
    tr, te = d[d.split == "train"], d[d.split == "test"]
    fl = flagged.loc[te.index].to_numpy()
    print("config used:", cfg)
    print("rows the strict rule leaves unstripped: train", int(flagged.loc[tr.index].sum()),
          "| test", int(fl.sum()), "of", len(te))
    for col in ("text", "current", "strict"):
        p = predict(tr, te, col, cfg)
        ok = p == te.label.to_numpy()
        print(f"{col:8s} macro-F1 {f1_score(te.label, p, average='macro'):.4f}"
              f" | accuracy on those rows {ok[fl].mean():.3f} | on all other rows {ok[~fl].mean():.3f}")


if __name__ == "__main__":
    main()
