"""Per-dataset scores of the combined MARBERT vs single-dataset MARBERT. Run: uv run python -m arabic_sentiment.combined_report"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def boot(y, p, n=500, seed=42):
    rng = np.random.default_rng(seed)
    s = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])


c = pd.read_parquet(ROOT / "results/preds/combined-marbert-s0.parquet")
rows = []
for d, g in c.groupby("dataset"):
    y, p = g.label.to_numpy(), g.pred.to_numpy()
    lo, hi = boot(y, p)
    single = "hard_notitle" if d == "hard_notitle" else d
    files = [f for f in glob.glob(str(ROOT / f"results/runs/{single}-marbert-s*.json")) if "tuned" not in f]
    s = np.mean([json.load(open(f))["test_macro_f1"] for f in files]) if files else float("nan")
    rows.append({"dataset": d, "n": len(g), "combined": round(f1_score(y, p, average="macro"), 4),
                 "ci95": f"[{lo:.3f}, {hi:.3f}]", "single_marbert": round(s, 4)})
out = pd.DataFrame(rows)
out["diff"] = (out.combined - out.single_marbert).round(4)
print(out.to_string(index=False))
