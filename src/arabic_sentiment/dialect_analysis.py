"""Per-dialect macro-F1 for encoders and ALLaM, with bootstrap CIs.

Run: uv run python -m arabic_sentiment.dialect_analysis
Dialect labels come from a third-party classifier (not human annotation), kept at conf >= 0.7.
Output: results/dialect_analysis.csv
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
MIN_N, MIN_CONF = 50, 0.7
ENCODERS = ["arabert", "marbert", "xlmr"]
DATASETS = ["labr", "hard", "hard_notitle", "astd", "ajgt"]


def f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)


def boot_ci(y, p, n=500, seed=42):
    rng = np.random.default_rng(seed)
    s = [f1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])


def cell(y, p):
    if len(y) < MIN_N:
        return f"small (n={len(y)})"
    lo, hi = boot_ci(y, p)
    return f"{f1(y, p):.3f} [{lo:.3f}, {hi:.3f}] n={len(y)}"


def encoder_preds(d, m):
    """Return text, label, pred (majority vote across seeds, ties go to the positive class)."""
    files = [f for f in glob.glob(str(ROOT / f"results/preds/{d}-{m}-s*.parquet")) if "tuned" not in f]
    if not files:
        return None
    frames = [pd.read_parquet(f)[["text", "label", "pred"]] for f in files]
    base = frames[0][["text", "label"]].copy()
    base["pred"] = (np.mean([f.pred.to_numpy() for f in frames], axis=0) >= 0.5).astype(int)
    return base


def allam_preds(d):
    f = ROOT / f"results/llm/allam-{d}-test-en.jsonl"
    if not f.exists():
        return None
    r = pd.DataFrame([json.loads(l) for l in open(f, encoding="utf-8")]).drop_duplicates("i")
    return r.rename(columns={"gold": "label", "pred": "pred"})[["text", "label", "pred"]]


def main():
    tags = pd.read_parquet(ROOT / "results/dialect_tags.parquet")
    tags = tags[tags.conf >= MIN_CONF]
    rows = []
    for d in DATASETS:
        t = tags[tags.dataset == ("hard" if d == "hard_notitle" else d)].drop_duplicates("text")
        systems = {m: encoder_preds(d, m) for m in ENCODERS}
        systems["allam"] = allam_preds(d)
        for name, p in systems.items():
            if p is None:
                continue
            m = p.merge(t[["text", "dialect"]], on="text", how="inner")
            if d == "ajgt":
                groups = {"all confident rows": m}
            elif d == "astd":
                groups = {g: m[m.dialect == g] for g in sorted(m.dialect.unique())}
            else:
                groups = {"MSA": m[m.dialect == "MSA"], "non-MSA": m[m.dialect != "MSA"]}
            groups["ALL (any confidence)"] = p
            for g, sub in groups.items():
                rows.append({"dataset": d, "system": name, "group": g,
                             "result": cell(sub.label.to_numpy(), sub.pred.to_numpy())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/dialect_analysis.csv", index=False)
    for d, g in out.groupby("dataset"):
        print("\n====", d)
        print(g.pivot(index="group", columns="system", values="result").to_string())


if __name__ == "__main__":
    main()
