"""Write a blind tagging sheet of the 61 MARBERT errors read in Chapter 14.
Run: uv run python -m arabic_sentiment.error_sheet
Output: data/processed/error_sheet.csv (not tracked). Fill the tag column before looking at any tags in the guide.
Tags: none (no clear sentiment) | mixed | gold_wrong | context (stance, sarcasm, knowledge) | dialect | other
"""
import pandas as pd

from arabic_sentiment.dialect_analysis import ROOT


def main():
    frames = []
    for d in ["ajgt", "astd", "labr"]:
        p = pd.read_parquet(ROOT / f"results/preds/{d}-marbert-s0.parquet")
        err = p[p.label != p.pred]
        e = err.sample(min(25, len(err)), random_state=0)
        frames.append(e.assign(dataset=d))
    out = pd.concat(frames)[["dataset", "text", "label", "pred"]]
    out["tag"] = ""
    out.to_csv(ROOT / "data/processed/error_sheet.csv", index=False, encoding="utf-8-sig")
    print(len(out), "rows:", out.groupby("dataset").size().to_dict())
    print(out.groupby("dataset").text.first().str[:40].to_dict())


if __name__ == "__main__":
    main()
