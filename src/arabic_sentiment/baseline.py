"""TF-IDF + logistic regression baseline.

Run: uv run python -m arabic_sentiment.baseline
Config is selected on validation macro-F1; test is scored once per dataset.
Output: results/baseline.json
"""
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
DATASETS = ["labr", "hard", "hard_notitle", "astd", "ajgt"]
TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
REPEAT = re.compile(r"(.)\1{2,}")


def normalize(t):
    t = TASHKEEL.sub("", t).replace("\u0640", "")
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي")
    return REPEAT.sub(r"\1\1", t)


def vectorizer(kind):
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000)


def bootstrap_ci(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(f1_score(y[i], p[i], average="macro"))
    return [round(float(x), 4) for x in np.percentile(scores, [2.5, 97.5])]


def run(name):
    df = pd.read_parquet(DATA / f"{name}.parquet")
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    best = None
    for norm, kind in product([False, True], ["word", "char"]):
        txt = {s: (d.text.map(normalize) if norm else d.text) for s, d in parts.items()}
        v = vectorizer(kind)
        X = {"train": v.fit_transform(txt["train"])}
        X["val"], X["test"] = v.transform(txt["val"]), v.transform(txt["test"])
        for C in (1, 10, 100):
            m = LogisticRegression(C=C, solver="liblinear", max_iter=1000).fit(X["train"], parts["train"].label)
            f1_val = f1_score(parts["val"].label, m.predict(X["val"]), average="macro")
            print(f"  {name} norm={norm} feat={kind} C={C} val macro-F1={f1_val:.4f}")
            if best is None or f1_val > best["val_macro_f1"]:
                best = dict(norm=norm, feat=kind, C=C, val_macro_f1=f1_val, model=m, Xtest=X["test"])
    y = parts["test"].label.to_numpy()
    p = best["model"].predict(best["Xtest"])
    return {
        "dataset": name,
        "config": {"normalize": best["norm"], "features": best["feat"], "C": best["C"]},
        "val_macro_f1": round(best["val_macro_f1"], 4),
        "test_macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        "test_macro_f1_ci95": bootstrap_ci(y, p),
        "test_accuracy": round(float(accuracy_score(y, p)), 4),
        "n_test": int(len(y)),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = []
    for name in DATASETS:
        print(name)
        out.append(run(name))
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    t = pd.DataFrame(out)
    t["config"] = t.config.map(lambda c: f"norm={c['normalize']} {c['features']} C={c['C']}")
    print("\n" + t[["dataset", "config", "val_macro_f1", "test_macro_f1", "test_macro_f1_ci95", "test_accuracy", "n_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
