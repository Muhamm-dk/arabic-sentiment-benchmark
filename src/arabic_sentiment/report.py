"""Build the master results table (test macro-F1) as markdown.

Run: uv run python -m arabic_sentiment.report
Output: results/results_table.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["labr", "hard", "hard_notitle", "astd", "ajgt"]
COLS = ["tfidf", "arabert", "marbert", "xlmr", "xlmr (tuned)"]


def label(run, model):
    tag = "-".join(run.split("-")[3:])
    return "xlmr (tuned)" if model == "xlmr" and tag == "tuned" else model


def main():
    rows = [json.load(open(f)) for f in (ROOT / "results/runs").glob("*.json")]
    df = pd.DataFrame(rows)
    df["col"] = [label(r, m) for r, m in zip(df.run, df.model)]
    g = df.groupby(["dataset", "col"]).test_macro_f1.agg(["mean", "std", "count"])
    base = {b["dataset"]: b for b in json.load(open(ROOT / "results/baseline.json"))}
    lines = ["| Dataset | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for d in ORDER:
        cells = [f"{base[d]['test_macro_f1']:.4f}"]
        for c in COLS[1:]:
            if (d, c) not in g.index:
                cells.append("-")
                continue
            m, sd, n = g.loc[(d, c)]
            cells.append(f"{m:.4f} ± {sd:.4f} (n={int(n)})" if n > 1 else f"{m:.4f}")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    out = "\n".join(lines)
    (ROOT / "results/results_table.md").write_text(out + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
