# Chapter 11: Few-shot LLM evaluation

## What

Score three hosted language models on the same test rows as the encoders, by showing each one eight labeled examples and asking for a one-word answer. No training is involved. Code: `src/arabic_sentiment/llm_eval.py` (one script) and `run_llm_queue.sh` (runs the five test sets for one model).

Models (all on Groq's free tier, run October 2026): `allam-2-7b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`.

## Why

- **The brief asked for it:** fine-tuned encoders against a few-shot Arabic-capable LLM. The interesting question is whether a general model with eight examples matches a 135M to 163M parameter encoder fine-tuned on thousands.
- **Why an API:** the models are too large for 8 GB of VRAM, and the free tier costs nothing.
- **Why these three:** ALLaM is an Arabic-specialist model (from SDAIA in Saudi Arabia, from memory, not checked here). gpt-oss-120b is a large general reasoning model and Qwen is a 27B general multilingual model. I have no further information about Qwen than its id. I picked them from the models the account listed. Speech and safety models and `gpt-oss-20b` were skipped.
- **Why not Jais** (the model named in the brief): about 13B parameters, so 4-bit loading would need roughly 8 to 9 GB by my estimate, which does not fit the laptop. It needed a Kaggle T4, and I could not verify that its custom model code runs under the current library. ALLaM filled the same role with much less risk. The Kaggle route was dropped, not tried.

## The protocol

| Decision | Choice | Reason |
|---|---|---|
| Format | Chat: one system instruction, then eight user and assistant message pairs, then the test text | Native format for chat models |
| Instruction | "Classify the sentiment of the Arabic text as positive or negative. Reply with exactly one word: positive or negative." | Short and unambiguous |
| Shots | 4 positive and 4 negative from the **train** split, random with seed 42, shuffled once, same eight for every row | Balanced, never from validation or test |
| Truncation | Each shot to 60 words, each test text to 200 words | Keeps prompts small under the token limits |
| Decoding | Temperature 0, output cap 16 tokens (ALLaM, Qwen) or 1,000 (gpt-oss) | Deterministic as far as the API allows |
| Reasoning | `reasoning_effort` "low" for gpt-oss, "none" for Qwen | So the model does not spend the budget on hidden reasoning. The parameters were tested in a probe |
| Parsing | Reply lowercased, stripped of spaces and `.!"'`, then exactly `positive` or `negative`; anything else is -1 | Strict |
| Invalid replies | Counted as wrong, rate reported | A refusal is a real failure |
| Retries | None for bad answers | At temperature 0 a retry gives the same answer |
| Rows | AJGT 180 and ASTD 242 (full test sets), LABR 300 and HARD 300 (seeded, label-stratified), the same 300 rows for `hard` and `hard_notitle` | Daily token quotas |
| Metric | Macro-F1 over labels 0 and 1, 1,000-resample bootstrap interval, plus accuracy and invalid rate | Same as every other system |

### Prompt language: a rule fixed in advance

English instruction against Arabic instruction, 60 AJGT validation tweets (stratified), Arabic chosen only if it wins by at least 5 points of macro-F1. Test data was not involved.

| Model | English | Arabic | Invalid (en / ar) | Decision |
|---|---|---|---|---|
| Qwen | 0.9000 | 0.8833 | 0% / 0% | English |
| ALLaM | 0.7725 | 0.7376 | 6.7% / 11.7% | English |
| gpt-oss | 0.8548 then 0.8810 (see below) | not run | 5.0% then 3.3% | English by default |

Weaknesses: 60 tweets give intervals of about ±8 points, so both differences are noise, which is why the rule asked for 5. The choice used Jordanian tweets only and was applied to reviews as well. For gpt-oss the Arabic prompt crashed twice (a server error, below), so English was applied as the default and never tested.

### gpt-oss output budget: one change on validation

The first validation run (300-token budget) had 3 empty generations of 60. Raising the budget to 1,000 gave 2 of 60 and a macro-F1 move from 0.8548 to 0.8810. One row is worth 1.7 points at that size, so I read it as no clear gain. I stopped tuning after that one change and kept 1,000 for all test runs.

## How: the code

### Building a request

```python
def make_shots(df):
    tr = df[df.split == "train"]
    shots = pd.concat([tr[tr.label == 1].sample(4, random_state=SEED),
                       tr[tr.label == 0].sample(4, random_state=SEED)])
    return shots.sample(frac=1, random_state=SEED)

def build_messages(system, shots, text):
    m = [{"role": "system", "content": system}]
    for _, r in shots.iterrows():
        m += [{"role": "user", "content": clip(r.text, SHOT_WORDS)},
              {"role": "assistant", "content": NAME[r.label]}]
    m.append({"role": "user", "content": clip(text, TEXT_WORDS)})
    return m
```

- Shots are drawn from the train split of the dataset being evaluated. For `hard`, the eight shots carry the leading score label; for `hard_notitle` they do not.
- The shot order was shuffled once. The same order is used every time, so order effects are not measured.
- `clip` keeps the first N whitespace-separated words.

### Row selection

```python
def pick_rows(df, split, n):
    d = df[df.split == split].reset_index(drop=True)
    if n and n < len(d):
        keep, _ = train_test_split(np.arange(len(d)), train_size=n,
                                   stratify=d.label, random_state=SEED)
        d = d.iloc[np.sort(keep)].reset_index(drop=True)
    return d
```

A seeded stratified sample, returned in the original order. `hard` and `hard_notitle` have identical labels, splits and row order, so they select the same 300 positions (Chapter 4).

### Calling the API

```python
def call(key, body):
    for attempt in range(8):
        resp = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            print("429 body:", resp.text[:300], ...)
            wait = float(resp.headers.get("retry-after", 20))
            if wait > 600 or "per day" in resp.text.lower():
                raise RateLimited(...)
            time.sleep(wait + 1)
            continue
        if resp.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 400 and "output_parse_failed" in resp.text:
            return resp
        raise RuntimeError(...)
```

- A 429 with a short wait is retried. A daily-token 429 raises `RateLimited`, which stops the run cleanly so it can be resumed after the refill.
- Server errors (5xx) are retried with a growing delay.
- A 400 with `output_parse_failed` (the provider could not extract an answer) is returned, and the caller stores it as an invalid reply.
- Up to 8 attempts. After that the script crashes (the original stall came from a missing daily-limit check, see below).
- The key comes from the `GROQ_API_KEY` environment variable and is never written to a file.

### The loop and resuming

```python
done = {}
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line); done[r["i"]] = r
with path.open("a", encoding="utf-8") as f:
    for i, r in rows.iterrows():
        if i in done:
            assert done[i]["text"] == r.text, "row mismatch ..."
            continue
        ...request...
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        done[i] = rec
        if int(resp.headers.get("x-ratelimit-remaining-tokens", 99999)) < 2500:
            time.sleep(seconds(resp.headers.get("x-ratelimit-reset-tokens")) + 0.5)
```

- Every reply is appended and flushed as it arrives, so an interrupted run loses nothing and rerunning the same command continues. The saved text is compared with the current row, so a changed `--n` fails instead of mixing sets.
- The remaining-tokens header triggers a proactive pause before the per-minute limit is hit.
- The file is keyed by row index. Two queues on one file produce duplicate lines (5 appeared once). Scores are unaffected, because the dict keeps one record per index.
- A summary file (`.json`) is written only when all rows are done.
- The summary date is the day the last row finished.

### Scoring

```python
def macro_f1(y, p):
    return float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0))
```

An invalid reply (-1) is a wrong prediction for its row. `labels=[0, 1]` keeps -1 from becoming a third class. `avg_in_tokens` in the summary includes zero tokens for failed rows, so it is slightly low where failures exist.

### The runner

```bash
for spec in "ajgt" "astd" "labr --n 300" "hard --n 300" "hard_notitle --n 300"; do
  uv run python -m arabic_sentiment.llm_eval --model $m --dataset $spec --split test --prompt en
done
```

Run as `nohup ./run_llm_queue.sh <model> > results/llm_<model>.log 2>&1 &`, one queue per model.

## Alternatives

None of these were run.

| Choice | Alternative |
|---|---|
| 8 random shots | Zero-shot, more shots, nearest-neighbour shots chosen per input, shots from the same domain |
| Same shot order | Several orders (few-shot results are known to depend on order, from memory) |
| Free-text one-word reply | Constrained decoding, scoring the probability of the two label words, JSON output |
| Direct answer | Chain-of-thought, self-consistency by voting |
| English instruction | Arabic (tested on validation for two models), Arabic label words |
| Hosted models | A local or Kaggle-hosted Jais or ALLaM with 4-bit quantization (no daily token cap) |
| Prompting only | Fine-tuning an LLM with LoRA |
| 300-row samples | Full test sets (needs paid quota or several more days) |
| Hard truncation at 200 words | Longer limit, or summaries |

## What went wrong

| Incident | Detail |
|---|---|
| Wrong quota plan | The probe showed per-minute and per-day request limits (ALLaM 7,000 requests a day, gpt-oss and Qwen 1,000) and I planned around those. The limit that bound was **200,000 tokens per day** for Qwen and gpt-oss, which first appeared in an error message: Qwen used 199,983 of 200,000. A 300-row LABR set alone costs about 200,000 tokens. gpt-oss also has **30 requests per minute** |
| A silent stall | The first run piped output through `grep`, which hid the progress lines. The script was waiting out per-minute windows, getting one request through at a time, and blocking again. Fix: stop on a daily-token message, and run without filters |
| Slow schedule | The test runs took about three days: ALLaM finished on day 1, gpt-oss and Qwen needed refills |
| gpt-oss empty generations | `output_parse_failed` with no text: 5.0% and 3.3% on validation, then 10.0% (AJGT) and 7.4% (ASTD) on test. A tiny curl test (5-token cap) returned an empty reply with the text in a `reasoning` field, so reasoning tokens count against the output budget. I did not establish why the failures persist at 1,000 tokens |
| ALLaM off-task replies | Replies such as "as an AI assistant I have no opinions" appeared for both positive and negative tweets. The 16-token cap cut them off (finish reason `length`). I first guessed the refusals were specific to negative tweets, and that was wrong |
| Invalid key | A call returned 401 after earlier successes. I did not establish why. A new key worked. A key was also pasted into a chat once and rotated. Keys now go through `read -rs` into an environment variable |
| Duplicate queue | A second queue on the same file, 5 duplicate lines. Rule: one queue per model, and never edit a result file while a queue runs |
| A rewrite during a write | A dedupe script ran while the queue was appending. The check afterwards found no gap |
| Partial results looked final | `compare.py` once printed HARD with n=1, because it read an unfinished file. A score is valid only when its summary file exists |
| Command-line corruption | Long pasted scripts were cut or spliced several times (Appendix A) |

## Numbers

Test macro-F1 (invalid replies counted as wrong, invalid rate in brackets). AJGT and ASTD use the full test sets, LABR and HARD 300-row samples.

| Dataset (n) | Qwen | gpt-oss | ALLaM |
|---|---|---|---|
| AJGT (180) | 0.8778 (0.0%) | 0.8301 (10.0%) | 0.8790 (3.3%) |
| ASTD (242) | 0.8966 (0.0%) | 0.8265 (7.4%) | 0.7522 (6.6%) |
| LABR (300) | 0.8797 (0.0%) | 0.8400 (0.0%) | 0.8453 (1.0%) |
| HARD (300) | 0.9265 (0.0%) | 0.8664 (1.0%) | 0.9266 (0.0%) |
| HARD no-descriptor (300) | 0.8622 (0.0%) | 0.7847 (1.7%) | 0.8759 (0.7%) |

Against the other systems on the same rows (encoders on ASTD and AJGT are 3-seed means, on LABR and HARD single seed):

| Dataset | TF-IDF | AraBERT | MARBERT | XLM-R | Best LLM |
|---|---|---|---|---|---|
| AJGT | 0.8722 | 0.9110 | 0.9277 | 0.8685 | Qwen 0.8778, ALLaM 0.8790 |
| ASTD | 0.7708 | 0.8665 | 0.8902 | 0.8367 | Qwen 0.8966 |
| LABR (300) | 0.8833 | 0.8733 | 0.9167 | 0.8699 | Qwen 0.8797 |
| HARD (300) | 0.9500 | 0.9533 | 0.9600 | 0.9533 | ALLaM 0.9266, Qwen 0.9265 |
| HARD no-descr. (300) | 0.8500 | 0.9267 | 0.9066 | 0.8933 | ALLaM 0.8759 |

Interval widths: about ±5 points on ASTD and AJGT, about ±4 on the 300-row sets (Chapter 6).

### What the numbers support

- **Qwen ties the best encoder on ASTD** (0.8966 against 0.8902). The difference is far inside the interval.
- **On AJGT the best encoder is likely ahead:** MARBERT leads Qwen by 5.0 points, at the edge of the interval.
- **ALLaM, the Arabic specialist, is no better than TF-IDF on ASTD** (0.7522 against 0.7708).
- **On reviews, the LLMs roughly match TF-IDF** and trail the encoders (LABR 300 rows: Qwen 0.8797 and TF-IDF 0.8833). On HARD without the descriptor the encoders (0.893 to 0.927) are ahead of Qwen and ALLaM (0.862, 0.876) by gaps of about 2 to 6 points, which a 300-row interval cannot separate for most pairs. gpt-oss at 0.7847 is clearly below the encoders.
- **The leak affects LLMs too.** Removing the descriptor costs Qwen 6.4 points, ALLaM 5.1 and gpt-oss 8.2 (Chapter 5). The shots on `hard` contain descriptors, which may teach the shortcut. I did not test that.

### What they do not support

- A ranking among the three LLMs on most datasets. Qwen's lead is clear on ASTD and weaker elsewhere.
- Any conclusion about LLMs in general. Three models, one prompt, one run each.
- That gpt-oss is a weak classifier. Much of its gap comes from invalid replies. A script below separates this out.

## Diagnostics script (provided, not yet run)

It separates invalid replies from wrong answers and shows label bias from the saved replies (the `.jsonl` files exist locally; they are not in git).

```bash
nano src/arabic_sentiment/llm_report.py
```

```python
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
```

```bash
uv run python -m py_compile src/arabic_sentiment/llm_report.py && uv run python -m arabic_sentiment.llm_report
```

Predictions made before running (rough arithmetic, so they can be wrong):
1. gpt-oss's score on valid replies only is about 0.90 or more on AJGT and ASTD (dividing its score by the share of valid rows suggests roughly 0.92 and 0.89), so its low numbers are mostly invalid replies.
2. ALLaM over-predicts positive: its predicted-positive share exceeds the true share, and its negative recall is the lowest of the three, on ASTD especially.
3. Qwen's predicted-positive share is near the true share.

## Not done

- A second prompt, more shots, nearest-neighbour shots, or several shot orders.
- Multiple runs per model, and checks of whether hosted models change between days (Qwen's `hard_notitle` rows span two days).
- Full test sets for LABR and HARD (the quota).
- Jais, or any model on Kaggle.
- A contamination check (these public datasets may be in the models' training data, and I cannot rule it out).
- A paired comparison of LLMs against encoders on the same rows (possible, since the rows are matched).
- Confidence or calibration for LLM answers.

## Check yourself

1. Why must few-shot examples come from the train split only?
2. Why were reasoning parameters set per model, and what went wrong with gpt-oss?
3. Why is an invalid reply counted as wrong, and how does `labels=[0, 1]` keep it from becoming a third class?
4. Why did the true quota limit differ from the one the probe showed, and what did the stall teach you about watching long jobs?
5. Why can the 300-row LABR and HARD numbers not be compared with the full-test table?
6. What could the descriptor in the `hard` shots do to an LLM?
7. Why was English chosen over Arabic, and what are two weaknesses of that choice?
8. Name three things that would make the LLM comparison fairer.
