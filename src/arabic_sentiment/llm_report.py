"""Per-model diagnostics from saved LLM replies. Run: uv run python -m arabic_sentiment.llm_report"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def f1(a, b):
    return f1_score(a, b, average="macro", labels=[0, 1], zero_division=0)


rows = []
for f in sorted((ROOT / "results/llm").glob("*-test-en.jsonl")):
    model, dataset = f.name.split("-")[0], f.name.split("-")[1]
    d = pd.DataFrame([json.loads(line) for line in open(f, encoding="utf-8")]).drop_duplicates("i")
    y, p = d.gold.to_numpy(), d.pred.to_numpy()
    ok = p != -1
    rows.append({
        "model": model, "dataset": dataset, "n": len(d),
        "invalid%": round(100 * (~ok).mean(), 1),
        "f1_all": round(f1(y, p), 4),
        "f1_valid_only": round(f1(y[ok], p[ok]), 4),
        "pred_pos%": round(100 * (p[ok] == 1).mean(), 1),
        "gold_pos%": round(100 * y.mean(), 1),
        "recall_neg": round(float(((p == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)), 3),
        "recall_pos": round(float(((p == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)), 3),
        "finish": dict(d.finish.value_counts()),
    })
print(pd.DataFrame(rows).to_string(index=False))
