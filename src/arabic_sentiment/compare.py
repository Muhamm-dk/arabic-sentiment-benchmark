"""LLM vs encoder comparison on identical rows. Run: uv run python -m arabic_sentiment.compare"""
import glob
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["ajgt", "astd", "labr", "hard", "hard_notitle"]
LLMS = ["qwen", "gptoss", "allam"]
ENCODERS = ["tfidf", "arabert", "marbert", "xlmr"]


def main():
    base = {b["dataset"]: b["test_macro_f1"] for b in json.load(open(ROOT / "results/baseline.json"))}
    rows = []
    for d in ORDER:
        ref = ROOT / f"results/llm/gptoss-{d}-test-en.jsonl"
        if not ref.exists():
            continue
        texts = {json.loads(l)["text"] for l in open(ref, encoding="utf-8")}
        row = {"dataset": d, "n": len(texts), "tfidf": base[d] if d in ("ajgt", "astd") else None}
        for m in ENCODERS:
            scores = []
            for f in glob.glob(str(ROOT / f"results/preds/{d}-{m}-s*.parquet")):
                if "tuned" in f:
                    continue
                p = pd.read_parquet(f)
                p = p[p.text.isin(texts)]
                if len(p) == 0 or len(p) != len(texts):
                    print(f"skip {Path(f).name}: matched {len(p)} of {len(texts)} rows")
                    continue
                scores.append(f1_score(p.label, p.pred, average="macro"))
            row[m] = round(sum(scores) / len(scores), 4) if scores else None
        for m in LLMS:
            s = ROOT / f"results/llm/{m}-{d}-test-en.json"
            if s.exists():
                r = json.load(open(s))
                row[m] = f"{r['macro_f1']:.4f} (inv {r['invalid_rate']:.1%})"
        rows.append(row)
    out = pd.DataFrame(rows).fillna("-")
    print(out.to_string(index=False))
    out.to_csv(ROOT / "results/llm_vs_encoders.csv", index=False)


if __name__ == "__main__":
    main()
