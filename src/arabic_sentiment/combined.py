"""Build the combined training set for the deployed model.
Run: uv run python -m arabic_sentiment.combined
Output: data/processed/combined.parquet (same columns as the single-dataset files)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARTS = ["labr", "hard_notitle", "astd", "ajgt"]
HARD_TRAIN_CAP = 20_000


def main():
    frames = []
    for d in PARTS:
        df = pd.read_parquet(ROOT / f"data/processed/{d}.parquet")
        if d == "hard_notitle":
            tr = df[df.split == "train"].sample(HARD_TRAIN_CAP, random_state=42)
            df = pd.concat([tr, df[df.split != "train"]])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["text", "split"])
    train_t = set(out[out.split == "train"].text)
    leak = out[(out.split != "train") & out.text.isin(train_t)]
    print("val/test rows whose text also appears in train:", len(leak))
    out = out.drop(leak.index)
    out.to_parquet(ROOT / "data/processed/combined.parquet", index=False)
    print(out.groupby(["split", "dataset"]).size().unstack(fill_value=0))
    print(out.groupby("split").label.mean().round(3))


if __name__ == "__main__":
    main()
