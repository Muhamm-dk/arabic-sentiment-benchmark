"""Tag test rows with a predicted dialect (third-party classifier). Run: uv run python -m arabic_sentiment.dialect
Output: results/dialect_tags.parquet with columns dataset, text, dialect, conf, dialect_qadi, conf_qadi
"""
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[2]
MAIN = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
SECOND = "oahmedd/MARBERTv2-Finetuned-on-5-Dialects-QADI"
DATASETS = ["labr", "hard", "astd", "ajgt"]  # hard_notitle shares rows with hard


def tag(model, texts):
    clf = pipeline("text-classification", model=model, device=0, truncation=True, max_length=128,
                   torch_dtype=torch.bfloat16)
    out = clf(texts, batch_size=64)
    del clf
    torch.cuda.empty_cache()
    return [o["label"] for o in out], [round(o["score"], 4) for o in out]


def main():
    frames = []
    for d in DATASETS:
        df = pd.read_parquet(ROOT / f"data/processed/{d}.parquet")
        t = df[df.split == "test"][["text"]].reset_index(drop=True)
        t["dataset"] = d
        frames.append(t)
    all_ = pd.concat(frames, ignore_index=True)
    texts = all_.text.tolist()
    all_["dialect"], all_["conf"] = tag(MAIN, texts)
    all_["dialect_qadi"], all_["conf_qadi"] = tag(SECOND, texts)
    all_.to_parquet(ROOT / "results/dialect_tags.parquet", index=False)
    print(all_.groupby(["dataset", "dialect"]).size().unstack(fill_value=0))
    hi = all_[all_.conf >= 0.7]
    print("\nshare with conf >= 0.7:\n", hi.groupby("dataset").size() / all_.groupby("dataset").size())


if __name__ == "__main__":
    main()
