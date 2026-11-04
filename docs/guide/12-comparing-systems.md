# Chapter 12: Comparing systems

## What

Putting TF-IDF, three encoders and three LLMs side by side without letting row sets, seed counts or sample sizes bias the picture. Four scripts do this:

| Script | Output | Compares | Rows |
|---|---|---|---|
| `report.py` | `results/results_table.md` | TF-IDF and encoders | full test sets |
| `compare.py` | `results/llm_vs_encoders.csv` | TF-IDF, encoders and LLMs | the rows the LLMs saw |
| `paired_all.py` | `results/paired.csv` | pairs of TF-IDF and encoders | full test sets |
| `paired_llm.py` | `results/paired_llm.csv` | each LLM against TF-IDF and MARBERT | the rows the LLM saw |

`paired_llm.py` is provided below and had not been run when this chapter was written.

## Why row matching matters

The LLM test sets are the full tests for AJGT and ASTD but 300-row samples for LABR and HARD (a quota decision, Chapter 11). A 300-row sample is not the same test as the full set. This table shows how much each system's score moves between them:

| Dataset | System | Full test | 300-row sample | Sample minus full |
|---|---|---|---|---|
| LABR | TF-IDF | 0.8155 | 0.8833 | +6.8 |
| LABR | AraBERT | 0.8629 | 0.8733 | +1.0 |
| LABR | MARBERT | 0.8799 | 0.9167 | +3.7 |
| LABR | XLM-R | 0.8480 | 0.8699 | +2.2 |
| HARD | TF-IDF | 0.9411 | 0.9500 | +0.9 |
| HARD | AraBERT | 0.9568 | 0.9533 | -0.4 |
| HARD | MARBERT | 0.9561 | 0.9600 | +0.4 |
| HARD | XLM-R | 0.9530 | 0.9533 | +0.0 |
| HARD, descriptor removed | TF-IDF | 0.8710 | 0.8500 | -2.1 |
| HARD, descriptor removed | AraBERT | 0.9067 | 0.9267 | +2.0 |
| HARD, descriptor removed | MARBERT | 0.9045 | 0.9066 | +0.2 |
| HARD, descriptor removed | XLM-R | 0.9009 | 0.8933 | -0.8 |

Three consequences:

1. **The LABR sample is easier for everyone, but not equally.** On the sample, TF-IDF (0.8833) scores above AraBERT (0.8733) and XLM-R (0.8699). On the full test set, AraBERT is ahead of TF-IDF by 4.7 points with an interval that excludes 0 ([+3.0, +6.4], Chapter 10). The 300-row table must not be used to compare encoders with TF-IDF.
2. **On HARD without the descriptor, the sample exaggerates the encoder advantage over TF-IDF:** AraBERT minus TF-IDF is +7.7 on the sample and +3.6 on the full set.
3. **An unusually large shift.** TF-IDF's +6.8 on the LABR sample is about 3.3 standard deviations of the simulated sampling noise (2.07 points). Random samples of the same shape matched it in 2 of 10,000 draws. It is a rare draw and I found no defect in how it was drawn (see the simulation below). The LLM comparisons on LABR are therefore unfavourable to the LLMs and are not used for any claim about LLMs against TF-IDF.

So every LLM comparison on LABR and HARD is a comparison on that particular sample. They stay in the README labelled as such, and no LLM claim in this guide relies on a LABR or HARD sample ranking for the LLMs against TF-IDF.

## How `compare.py` matches rows

```python
ref = ROOT / f"results/llm/gptoss-{d}-test-en.jsonl"
texts = {json.loads(l)["text"] for l in open(ref, encoding="utf-8")}
...
for f in glob.glob(str(ROOT / f"results/preds/{d}-{m}-s*.parquet")):
    if not re.fullmatch(rf"{d}-{m}-s\d+(-n\d+)?\.parquet", Path(f).name):
        continue
    p = pd.read_parquet(f)
    p = p[p.text.isin(texts)]
    if len(p) == 0 or len(p) != len(texts):
        print(f"skip {Path(f).name}: matched {len(p)} of {len(texts)} rows")
        continue
    scores.append(f1_score(p.label, p.pred, average="macro"))
row[m] = round(sum(scores) / len(scores), 4) if scores else None
```

- The row set comes from one LLM's saved replies (gpt-oss). Every encoder's saved test predictions are filtered to those texts, and a file whose row count does not match is skipped with a message.
- Encoder scores are means over the seeds found (3 on ASTD and AJGT, 1 elsewhere). LLM scores come from each model's summary file (one run).
- Rows are matched by text. That works because texts are unique within a dataset after deduplication (Chapter 4).
- The filename pattern (Chapter 9) keeps ablation runs out.

Weaknesses, from reading the code:

1. It assumes the other two LLMs used the same rows as gpt-oss. They should (same function, same seed), but it is not checked.
2. A skipped seed lowers the seed count silently in the table (only the printed message says so).
3. The table has no intervals. They exist in the summary files and in `paired.csv`.
4. LLM cells are strings with the invalid rate included, so the CSV cannot be sorted numerically.
5. For TF-IDF on AJGT and ASTD the code first reads the number from `baseline.json` and then overwrites it with the value recomputed from predictions. They are identical (checked in Chapter 7).
6. `report.py`'s file-name pattern also accepts the combined model's run (`combined-marbert-s0`). The table ignores it because the dataset is not in its list.

## Paired comparisons

`paired_all.py` and its results are in Chapter 10. The points that matter for this chapter:

- Paired intervals are narrower than the intervals of two separate runs, because the same hard rows hurt both systems (on LABR, MARBERT against AraBERT: paired half-width about 1.3 points against about 1.8 for combined separate intervals).
- The intervals cover test-row sampling only. Training-seed noise is absent, and it is large relative to some gaps (AraBERT on ASTD has a standard deviation of 3.2 points across seeds).
- Six of 30 intervals have a lower bound under 0.2 points, and no correction for multiple comparisons was applied.

An improvement not made: resample rows once and average the difference over the three seeds, which tests the seed-averaged gap on ASTD and AJGT directly.

## LLMs against TF-IDF and MARBERT: `paired_llm.py`

Positive differences mean the LLM is ahead. Invalid replies count as wrong. The macro-F1 here is written out explicitly (unlike `paired_all.py`'s shortcut), because an invalid reply is neither 0 nor 1: it is never a true or false positive, and it is a miss for its true class.

```python
"""Paired bootstrap of each LLM against TF-IDF and MARBERT on the identical rows the LLM saw.
Positive difference means the LLM is ahead. Invalid LLM replies count as wrong.
Run: uv run python -m arabic_sentiment.paired_llm
Output: results/paired_llm.csv
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "results/preds"
DATASETS = ["ajgt", "astd", "labr", "hard", "hard_notitle"]
LLMS = ["qwen", "gptoss", "allam"]
BASES = ["tfidf", "marbert"]
N_BOOT, SEED = 2000, 42


def f1_class(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def macro_f1(y, p):
    pos = f1_class(np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p != 1)))
    neg = f1_class(np.sum((y == 0) & (p == 0)), np.sum((y == 1) & (p == 0)), np.sum((y == 0) & (p != 0)))
    return (pos + neg) / 2


def find(d, m, s):
    pat = re.compile(rf"{d}-{m}-s{s}(-n\d+)?\.parquet")
    hits = [f for f in PREDS.glob(f"{d}-{m}-s{s}*.parquet") if pat.fullmatch(f.name)]
    return hits[0] if hits else None


def llm_rows(model, d):
    f = ROOT / f"results/llm/{model}-{d}-test-en.jsonl"
    r = pd.DataFrame([json.loads(line) for line in open(f, encoding="utf-8")]).drop_duplicates("i")
    return r[["text", "gold", "pred"]].rename(columns={"gold": "label", "pred": "pred_llm"})


def main():
    rows = []
    for d in DATASETS:
        for llm in LLMS:
            L = llm_rows(llm, d)
            for base in BASES:
                res = {}
                for s in ((0,) if base == "tfidf" else (0, 1, 2)):
                    f = find(d, base, s)
                    if f is None:
                        continue
                    B = pd.read_parquet(f)[["text", "pred"]].rename(columns={"pred": "pred_base"})
                    m = L.merge(B, on="text")
                    assert len(m) == len(L), f"{d} {llm} vs {base} seed {s}: matched {len(m)} of {len(L)} rows"
                    y, pl, pb = m.label.to_numpy(), m.pred_llm.to_numpy(), m.pred_base.to_numpy()
                    res[s] = (macro_f1(y, pl) - macro_f1(y, pb), y, pl, pb)
                if 0 not in res:
                    continue
                diff0, y, pl, pb = res[0]
                rng = np.random.default_rng(SEED)
                boots = np.empty(N_BOOT)
                for k in range(N_BOOT):
                    i = rng.integers(0, len(y), len(y))
                    boots[k] = macro_f1(y[i], pl[i]) - macro_f1(y[i], pb[i])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({"dataset": d, "llm": llm, "base": base, "n": len(y), "invalid": int((pl == -1).sum()),
                             "diff_s0": round(diff0, 4), "ci95": f"[{lo:+.4f}, {hi:+.4f}]",
                             "excludes_0": bool(lo > 0 or hi < 0),
                             "diff_s1": round(res[1][0], 4) if 1 in res else None,
                             "diff_s2": round(res[2][0], 4) if 2 in res else None})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/paired_llm.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
```

Run it with `uv run python -m arabic_sentiment.paired_llm` (CPU, one or two minutes). The seed columns are for MARBERT only (TF-IDF has one run). The LLM rows come from the saved replies, which exist locally but are not in git.

Predictions committed before running (rough, from the scores above, so they can be wrong):

1. Qwen against MARBERT on ASTD: interval includes 0 (0.8966 against 0.8989 for seed 0).
2. Qwen against TF-IDF on ASTD: excludes 0, with Qwen ahead (+12.6 points on the same 242 rows).
3. ALLaM against TF-IDF on ASTD: includes 0 (0.7522 against 0.7708).
4. gpt-oss against MARBERT on HARD without the descriptor: excludes 0, with gpt-oss behind (0.7847 against 0.9066).
5. Every LLM against TF-IDF on AJGT: includes 0.

## LLMs against TF-IDF and MARBERT (measured)

`paired_llm.py` on the identical rows each LLM saw. Cells show the seed-0 difference in points (LLM minus the other system), the 95% paired interval, and `*` when the interval excludes 0. Invalid replies count as wrong. Counts of invalid replies are in `results/paired_llm.csv`.

LLM minus TF-IDF:

| Dataset | Qwen | gpt-oss | ALLaM |
|---|---|---|---|
| AJGT (180) | +0.6 [-5.5, +6.9] | -4.2 [-11.1, +2.5] | +0.7 [-5.2, +6.9] |
| ASTD (242) | +12.6 [+6.7, +18.8] * | +5.6 [-0.9, +12.3] | -1.9 [-9.1, +5.8] |
| LABR (300-row sample) | -0.4 [-4.8, +4.0] | -4.3 [-9.0, +0.6] | -3.8 [-8.5, +1.6] |
| HARD (300-row sample) | -2.4 [-5.3, +0.6] | -8.4 [-12.5, -4.5] * | -2.3 [-5.3, +0.3] |
| HARD no-descriptor (300-row sample) | +1.2 [-3.2, +5.6] | -6.5 [-11.8, -1.5] * | +2.6 [-1.9, +6.9] |

LLM minus MARBERT:

| Dataset | Qwen | gpt-oss | ALLaM |
|---|---|---|---|
| AJGT (180) | -3.9 [-9.4, +1.2] | -8.7 [-14.3, -3.4] * | -3.8 [-9.0, +1.3] |
| ASTD (242) | -0.2 [-3.5, +3.1] | -7.2 [-12.0, -2.5] * | -14.7 [-20.2, -8.9] * |
| LABR (300-row sample) | -3.7 [-7.8, +0.3] | -7.7 [-12.0, -3.3] * | -7.1 [-11.8, -2.3] * |
| HARD (300-row sample) | -3.4 [-6.1, -0.7] * | -9.4 [-12.9, -6.0] * | -3.3 [-6.0, -0.7] * |
| HARD no-descriptor (300-row sample) | -4.4 [-8.2, -0.7] * | -12.2 [-17.1, -7.7] * | -3.1 [-6.7, +0.6] |

Differences across MARBERT's three seeds on the tweet sets (point differences, no intervals):

| Dataset | LLM | MARBERT seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| AJGT (180) | Qwen | -3.9 | -6.1 | -5.0 |
| AJGT (180) | gpt-oss | -8.7 | -10.9 | -9.8 |
| AJGT (180) | ALLaM | -3.8 | -6.0 | -4.9 |
| ASTD (242) | Qwen | -0.2 | +0.4 | +1.7 |
| ASTD (242) | gpt-oss | -7.2 | -6.6 | -5.3 |
| ASTD (242) | ALLaM | -14.7 | -14.0 | -12.8 |

Reading:

- Qwen is the only LLM clearly above TF-IDF anywhere (ASTD, +12.6). gpt-oss is below TF-IDF on both HARD variants.
- Qwen is not distinguishable from MARBERT on ASTD (-0.2 points, interval [-3.5, +3.1]; -0.2, +0.5 and +1.7 across seeds). The interval rules out gaps beyond about 3.5 points and nothing smaller.
- gpt-oss is below MARBERT on all five test sets. It has 18 invalid replies on AJGT and 18 on ASTD, so part of that gap is invalid replies.
- ALLaM is below MARBERT on ASTD, LABR and raw HARD, and not distinguishable on AJGT and HARD without the descriptor.
- Qwen is below MARBERT on both HARD variants. On AJGT and LABR the intervals include 0, although all three AJGT seeds are negative.
- TF-IDF's score on the LABR sample is 6.8 points above its full-test score, which is unfavourable to the LLMs. On HARD without the descriptor it is 2.1 points below, which favours them.
- Thirty intervals were examined without correction for multiple comparisons. LLM runs are single runs.
- All five predictions made before running held.

## Check the LABR sample

This asks whether the 300 sampled rows differ from the other 1,983 test rows in a way the label balance does not show.

```bash
uv run python - <<'EOF'
import json, pandas as pd
d = pd.read_parquet("data/processed/labr.parquet").query("split == 'test'").copy()
s = {json.loads(l)["text"] for l in open("results/llm/qwen-labr-test-en.jsonl", encoding="utf-8")}
d["in_sample"] = d.text.isin(s)
d["words"] = d.text.str.split().str.len()
print(d.groupby("in_sample").agg(n=("text", "size"), pos=("label", "mean"), words_mean=("words", "mean"), words_median=("words", "median")))
EOF
```

Result of the first check, label balance and length: the sample has about the same positive share as the other 1,983 rows (50.7% against 50.5%) and a shorter mean length (53.7 against 59.9 words) but a longer median (32 against 30). Neither explains a shift of this size. Result of the simulation (`sample_check.py`, 10,000 random label-stratified draws of 152 positive and 148 negative rows from the 2,283 test rows): TF-IDF's observed shift of +6.8 points was matched or exceeded in 2 of 10,000 draws (simulated standard deviation 2.07). MARBERT's +3.7 was matched or exceeded in 1.5% of draws (sd 1.77), XLM-R's +2.2 in 11.8% (sd 1.94), AraBERT's +1.0 in 28.4% (sd 1.86). No draw matched both the TF-IDF and the MARBERT shift, which is implied by the TF-IDF figure. Reading: the sample is a rare draw, not a demonstrated defect. I found nothing wrong with how it was drawn (seeded, label-stratified random sample), but a 2-in-10,000 event among four systems examined cannot be fully excluded as a problem I did not find. The shifts differ by system, so the sample is not simply easier for everyone by a fixed amount.

## What this guide claims after the comparison

| Claim | Evidence | Strength |
|---|---|---|
| Fine-tuned encoders beat TF-IDF on LABR, HARD and HARD without the descriptor | Paired intervals exclude 0 for all three encoders | Strong, test-row sampling only. Single seed |
| AraBERT and MARBERT beat TF-IDF on ASTD | Intervals exclude 0, positive in all three seeds | Strong |
| XLM-R beats TF-IDF on ASTD | Lower bound +0.09 points, positive in all seeds | Weak |
| On AJGT the encoders' gains are suggestive only | Intervals include 0, three seeds positive for AraBERT and MARBERT, mixed for XLM-R | Suggestive |
| MARBERT is ahead of AraBERT on LABR | +1.7, [+0.4, +3.1], one seed | Moderate |
| MARBERT is level with AraBERT on HARD, ASTD and AJGT | Intervals include 0 | Not separable |
| AraBERT and MARBERT are slightly ahead of XLM-R | LABR, ASTD (MARBERT), HARD under 1 point | Moderate on LABR and ASTD, weak on HARD |
| The HARD descriptor inflates every system | Chapter 5 | Strong |
| Qwen is not distinguishable from MARBERT on ASTD | Paired difference -0.2 points, interval [-3.5, +3.1]; -0.2, +0.5, +1.7 across seeds | Moderate: rules out gaps beyond about 3.5 points on 242 tweets |
| LLM comparisons on LABR and HARD | 300-row samples with unexplained sampling differences | Weak |

## Not done

- A seed-pooled paired test, and correction for multiple comparisons.
- Repeated LLM runs, other prompts, or LLMs on the full LABR and HARD test sets.
- Comparing costs and speed. Measured only for the deployed encoder: about 0.9 s round trip (Chapter 18). LLM latency and cost per request were not recorded, so no claim is made about them.
- Calibration (do confidence scores mean what they say?).

## Check yourself

1. Why does the 300-row LABR sample not support a TF-IDF-against-encoder comparison, and what does the full test set show instead?
2. What does a paired bootstrap use that two separate intervals do not, and what noise does it still miss?
3. Six of the 30 intervals have lower bounds under 0.2 points. How should you word those results?
4. Why does `paired_llm.py` write its own macro-F1 instead of reusing the shortcut from `paired_all.py`?
5. On ASTD, AraBERT minus XLM-R is +4.5, +6.3 and -1.8 over three seeds. What does that tell you about a single-seed paired result?
6. Name two weaknesses in how `compare.py` builds its table.
7. The three seeds on AJGT all favour MARBERT over TF-IDF but the interval for seed 0 includes 0. Why is agreement across seeds not a confirmation?
