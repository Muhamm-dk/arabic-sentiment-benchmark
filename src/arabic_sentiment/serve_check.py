"""Does the API's code path reproduce the saved test predictions of the combined model?
Run (deployed path, CPU): CUDA_VISIBLE_DEVICES= uv run python -m arabic_sentiment.serve_check
Needs outputs/combined-marbert-s0 and results/preds/combined-marbert-s0.parquet.
"""
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from arabic_sentiment.paired_all import macro_f1
from arabic_sentiment.serve.app import MAX_BATCH, MAX_CHARS, app, classify, state

ROOT = Path(__file__).resolve().parents[2]


def main():
    comb = pd.read_parquet(ROOT / "results/preds/combined-marbert-s0.parquet")
    too_long = comb[comb.text.str.len() > MAX_CHARS]
    print(f"test rows over {MAX_CHARS} characters (the API answers 422): {len(too_long)} of {len(comb)}",
          too_long.dataset.value_counts().to_dict())
    sample = pd.concat([g.sample(min(300, len(g)), random_state=0) for _, g in comb.groupby("dataset")],
                       ignore_index=True)
    texts = sample.text.tolist()
    out = []
    with TestClient(app):  # runs the lifespan, which loads the model
        device = state["device"]
        for i in range(0, len(texts), MAX_BATCH):
            out += classify(texts[i:i + MAX_BATCH])
    sample["api_label"] = [int(o["label"] == "positive") for o in out]
    sample["api_prob"] = [o["prob_positive"] for o in out]
    sample["api_trunc"] = [o["truncated"] for o in out]
    diff = (sample.api_prob - sample.prob_pos).abs()
    dis = sample.api_label != sample.pred
    print(f"device {device} | rows {len(sample)} | label disagreements {int(dis.sum())} ({dis.mean():.2%})"
          f" | probability difference: mean {diff.mean():.5f}, max {diff.max():.4f}")
    if dis.any():
        print(sample[dis][["dataset", "label", "prob_pos", "api_prob"]].round(3).to_string(index=False))
    for d, g in sample.groupby("dataset"):
        y = g.label.to_numpy()
        print(f"{d:13s} n={len(g)} | macro-F1 saved {macro_f1(y, g.pred.to_numpy()):.4f}"
              f" | via API code {macro_f1(y, g.api_label.to_numpy()):.4f} | truncated {int(g.api_trunc.sum())}")


if __name__ == "__main__":
    main()
