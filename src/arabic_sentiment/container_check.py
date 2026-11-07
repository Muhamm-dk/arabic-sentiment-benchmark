"""End-to-end check of a running API against the saved predictions of the combined model.
Run: uv run python -m arabic_sentiment.container_check http://localhost:8000 [key_file]
     uv run python -m arabic_sentiment.container_check https://<azure host> ~/.sentiment_api_key
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]


def main():
    url = sys.argv[1].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if len(sys.argv) > 2:
        headers["X-API-Key"] = Path(sys.argv[2]).expanduser().read_text().strip()
    comb = pd.read_parquet(ROOT / "results/preds/combined-marbert-s0.parquet")
    comb = comb[comb.text.str.len() <= 5000]
    sample = pd.concat([g.sample(min(60, len(g)), random_state=1) for _, g in comb.groupby("dataset")],
                       ignore_index=True)
    t0 = time.time()
    print("health:", requests.get(url + "/health", timeout=300).json(), f"({time.time() - t0:.1f} s)")
    out = []
    for i in range(0, len(sample), 32):
        r = requests.post(url + "/predict_batch", json={"texts": sample.text.iloc[i:i + 32].tolist()},
                          headers=headers, timeout=300)
        r.raise_for_status()
        out += r.json()["results"]
    sample["api_label"] = [int(o["label"] == "positive") for o in out]
    sample["api_prob"] = [o["prob_positive"] for o in out]
    dis = sample.api_label != sample.pred
    diff = (sample.api_prob - sample.prob_pos).abs()
    print(f"rows {len(sample)} | label disagreements {int(dis.sum())} | probability difference: "
          f"mean {diff.mean():.5f}, max {diff.max():.4f} | model_version {out[0]['model_version']}"
          f" | total {time.time() - t0:.1f} s")
    if dis.any():
        print(sample[dis][["dataset", "label", "prob_pos", "api_prob"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
