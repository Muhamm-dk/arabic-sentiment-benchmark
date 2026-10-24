"""Few-shot sentiment classification with LLMs on Groq. Resumable: replies are appended to a jsonl file.

Run: GROQ_API_KEY=... uv run python -m arabic_sentiment.llm_eval --model qwen --dataset ajgt --split val --n 60 --prompt en
Rows: fixed seeded stratified subsample of the split (identical rows for hard and hard_notitle).
Shots: 4 positive and 4 negative from the train split only, fixed seed.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = {  # key: (model id, extra parameters, max output tokens)
    "allam": ("allam-2-7b", {}, 16),
    "gptoss": ("openai/gpt-oss-120b", {"reasoning_effort": "low"}, 1000),
    "qwen": ("qwen/qwen3.8-27b", {"reasoning_effort": "none"}, 16),
}
SYSTEMS = {
    "en": "Classify the sentiment of the Arabic text as positive or negative. Reply with exactly one word: positive or negative.",
    "ar": "صنف مشاعر النص العربي التالي على أنها إيجابية أو سلبية. أجب بكلمة واحدة فقط: positive أو negative.",
}
NAME = {1: "positive", 0: "negative"}
SEED = 42
SHOT_WORDS, TEXT_WORDS = 60, 200


class RateLimited(Exception):
    pass


def clip(text, n):
    return " ".join(text.split()[:n])


def parse_reply(r):
    r = (r or "").strip().lower().strip(".!\"' \n")
    return {"positive": 1, "negative": 0}.get(r, -1)  # -1 = unparseable


def seconds(s):
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(v) * units[u] for v, u in re.findall(r"([\d.]+)(ms|h|m|s)", s or ""))


def pick_rows(df, split, n):
    d = df[df.split == split].reset_index(drop=True)
    if n and n < len(d):
        keep, _ = train_test_split(np.arange(len(d)), train_size=n, stratify=d.label, random_state=SEED)
        d = d.iloc[np.sort(keep)].reset_index(drop=True)
    return d


def make_shots(df):
    tr = df[df.split == "train"]
    shots = pd.concat([tr[tr.label == 1].sample(4, random_state=SEED), tr[tr.label == 0].sample(4, random_state=SEED)])
    return shots.sample(frac=1, random_state=SEED)


def build_messages(system, shots, text):
    m = [{"role": "system", "content": system}]
    for _, r in shots.iterrows():
        m += [{"role": "user", "content": clip(r.text, SHOT_WORDS)}, {"role": "assistant", "content": NAME[r.label]}]
    m.append({"role": "user", "content": clip(text, TEXT_WORDS)})
    return m


def call(key, body):
    for attempt in range(8):
        resp = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            try:
                wait = float(resp.headers.get("retry-after", 20))
            except ValueError:
                wait = 20.0
            if wait > 600:
                raise RateLimited(f"quota exhausted, retry in {wait / 60:.0f} min: {resp.text[:200]}")
            time.sleep(wait + 1)
            continue
        if resp.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 400 and "output_parse_failed" in resp.text:
            return resp  # model produced nothing parseable: recorded as an invalid reply, not a crash
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("too many retries")


def macro_f1(y, p):
    return float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0))


def bootstrap_ci(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    s = [macro_f1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(x), 4) for x in np.percentile(s, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--prompt", default="en", choices=list(SYSTEMS))
    ap.add_argument("--max_requests", type=int, default=950)
    a = ap.parse_args()

    key = os.environ["GROQ_API_KEY"]
    model_id, extra, max_out = MODELS[a.model]
    df = pd.read_parquet(ROOT / f"data/processed/{a.dataset}.parquet")
    rows, shots = pick_rows(df, a.split, a.n), make_shots(df)
    tag = f"{a.model}-{a.dataset}-{a.split}-{a.prompt}"
    path = ROOT / f"results/llm/{tag}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["i"]] = r

    sent = 0
    with path.open("a", encoding="utf-8") as f:
        for i, r in rows.iterrows():
            if i in done:
                assert done[i]["text"] == r.text, "row mismatch with saved results, delete the jsonl or keep --n unchanged"
                continue
            if sent >= a.max_requests:
                print("request budget reached for this call")
                break
            body = {"model": model_id, "messages": build_messages(SYSTEMS[a.prompt], shots, r.text),
                    "temperature": 0, "max_completion_tokens": max_out, **extra}
            try:
                resp = call(key, body)
            except RateLimited as e:
                print(e)
                break
            if resp.status_code == 400:
                rec = {"i": int(i), "text": r.text, "gold": int(r.label), "reply": None, "pred": -1,
                       "finish": "output_parse_failed", "in_tok": 0, "out_tok": 0}
            else:
                j = resp.json()
                reply = j["choices"][0]["message"].get("content")
                rec = {"i": int(i), "text": r.text, "gold": int(r.label), "reply": reply, "pred": parse_reply(reply),
                       "finish": j["choices"][0]["finish_reason"],
                       "in_tok": j["usage"]["prompt_tokens"], "out_tok": j["usage"]["completion_tokens"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[i] = rec
            sent += 1
            if sent % 50 == 0:
                print(f"{tag}: {len(done)}/{len(rows)}")
            if int(resp.headers.get("x-ratelimit-remaining-tokens", 99999)) < 2500:
                time.sleep(seconds(resp.headers.get("x-ratelimit-reset-tokens")) + 0.5)

    if len(done) < len(rows):
        print(f"{tag}: incomplete, {len(done)}/{len(rows)} done. Rerun the same command to resume.")
        return
    y = np.array([done[i]["gold"] for i in range(len(rows))])
    p = np.array([done[i]["pred"] for i in range(len(rows))])
    res = {"tag": tag, "model_id": model_id, "date": time.strftime("%Y-%m-%d"), "prompt": a.prompt, "shots": len(shots),
           "n": int(len(y)), "macro_f1": round(macro_f1(y, p), 4), "macro_f1_ci95": bootstrap_ci(y, p),
           "accuracy": round(float((y == p).mean()), 4), "invalid_rate": round(float((p == -1).mean()), 4),
           "avg_in_tokens": round(float(np.mean([done[i]["in_tok"] for i in range(len(rows))])))}
    (ROOT / f"results/llm/{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
