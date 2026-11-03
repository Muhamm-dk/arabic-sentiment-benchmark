# Chapter 10: Encoder results

## What

Everything the three fine-tuned encoders produced, how to read it, and what it does and does not support. Data comes from `results/runs/*.json` (one file per run). The training code is Chapter 9, and the rules for reading intervals are Chapter 6.

## The runs

| Group | Datasets | Models | Seeds | Settings |
|---|---|---|---|---|
| Stage 1 | AJGT, ASTD | AraBERT, MARBERT, XLM-R | 3 | 5 epochs, batch 32, lr 2e-5 (18 runs; MARBERT AJGT seed 0 was the smoke test with the same settings) |
| Stage 2 | LABR | same | 1 | 4 epochs, XLM-R at batch 16 with accumulation 2 |
| Stage 3 | HARD, HARD no-descriptor | same | 1 | 3 epochs, a fixed 20,000-review training subsample, XLM-R at batch 16 with accumulation 2 |
| XLM-R tuned | AJGT, ASTD | XLM-R | 3 | lr 3e-5, 10 epochs, batch 16 (6 runs) |
| Ablation | LABR | MARBERT | 1 | max length 128 instead of 256 |
| Deployed model | combined | MARBERT | 1 | Chapter 15 |

33 runs make up the main tables. `results/preds` holds 40 prediction files: those 33, the ablation, the combined model, and 5 TF-IDF files.

## The main table

Macro-F1 on the full test splits. ASTD and AJGT are means over 3 seeds, the rest single seed.

| Dataset (test n) | TF-IDF | AraBERT | MARBERT | XLM-R | XLM-R tuned |
|---|---|---|---|---|---|
| LABR (2,283) | 0.8155 | 0.8629 | **0.8799** | 0.8480 | - |
| HARD, as released (10,707) | 0.9411 | **0.9568** | 0.9561 | 0.9530 | - |
| HARD, descriptor removed (10,707) | 0.8710 | **0.9067** | 0.9045 | 0.9009 | - |
| ASTD (242) | 0.7708 | 0.8665 ± 0.032 | **0.8902 ± 0.010** | 0.8367 ± 0.011 | 0.8067 ± 0.008 |
| AJGT (180) | 0.8722 | 0.9110 ± 0.011 | **0.9277 ± 0.011** | 0.8685 ± 0.009 | 0.9037 ± 0.006 |

### Gain over TF-IDF, in points

| Dataset | AraBERT | MARBERT | XLM-R |
|---|---|---|---|
| LABR | +4.7 | +6.4 | +3.3 |
| HARD, as released | +1.6 | +1.5 | +1.2 |
| HARD, descriptor removed | +3.6 | +3.4 | +3.0 |
| ASTD | +9.6 | +11.9 | +6.6 |
| AJGT | +3.9 | +5.5 | -0.4 |

### Ranking among encoders, in points

| Dataset | MARBERT minus AraBERT | MARBERT minus XLM-R |
|---|---|---|
| LABR | +1.7 | +3.2 |
| HARD, as released | -0.1 | +0.3 |
| HARD, descriptor removed | -0.2 | +0.4 |
| ASTD | +2.4 | +5.3 |
| AJGT | +1.7 | +5.9 |

## Every run

Test macro-F1 (best epoch in brackets, chosen on validation).

| Dataset | Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| AJGT | AraBERT | 0.9110 (2) | 0.9221 (2) | 0.8999 (1) |
| AJGT | MARBERT | 0.9166 (3) | 0.9389 (4) | 0.9277 (2) |
| AJGT | XLM-R | 0.8778 (5) | 0.8667 (3) | 0.8611 (4) |
| AJGT | XLM-R tuned | 0.9000 (9) | 0.9111 (9) | 0.8999 (6) |
| ASTD | AraBERT | 0.8786 (3) | 0.8904 (2) | 0.8306 (1) |
| ASTD | MARBERT | 0.8989 (2) | 0.8921 (2) | 0.8797 (1) |
| ASTD | XLM-R | 0.8340 (5) | 0.8278 (5) | 0.8483 (4) |
| ASTD | XLM-R tuned | 0.7981 (2) | 0.8095 (2) | 0.8126 (3) |

Single-seed runs (best epoch in brackets): LABR AraBERT 0.8629 (1 of 4), MARBERT 0.8799 (4), XLM-R 0.8480 (4). HARD AraBERT 0.9568 (3 of 3), MARBERT 0.9561 (2), XLM-R 0.9530 (3). HARD no-descriptor AraBERT 0.9067 (2), MARBERT 0.9045 (3), XLM-R 0.9009 (3).

## Reading the results

### What the numbers support

Using the Chapter 6 rules (a gap below half the interval width is noise; on ASTD and AJGT only about 5 points or more count as likely real):

1. **AraBERT and MARBERT beat TF-IDF on every dataset.** The clearest case is ASTD (+9.6 and +11.9, far outside TF-IDF's interval of about ±5.9). On LABR (+4.7, +6.4) the gaps exceed the interval width of 3.2. On HARD without the descriptor (+3.6, +3.4) they exceed the interval width of 1.2. On AJGT only MARBERT's +5.5 reaches the 5-point rule, and I call it likely, not proven.
2. **XLM-R does not clearly beat TF-IDF on tweets.** On AJGT it ties (-0.4). On ASTD +6.6 is probably real. Not on HARD: its +1.2 on the raw text is real at that test size, though small.
3. **The three encoders are not separable on HARD.** Gaps of 0.1 to 0.4 points on 10,707 reviews are inside an interval of about ±0.6, with one seed each.
4. **MARBERT leads on LABR, ASTD and AJGT, but the gaps over AraBERT (1.7, 2.4, 1.7 points) are not proven.** MARBERT over XLM-R (3.2, 5.3, 5.9) is more convincing, though XLM-R's tuned configuration narrows the AJGT gap to 2.4.
5. **HARD without the descriptor is the honest HARD number.** All three encoders gain about 3 to 3.6 points over TF-IDF there, against 1.2 to 1.6 on the raw text (Chapter 5).

### What it does not support

- Any ranking between MARBERT and AraBERT. The pattern (MARBERT ahead on three datasets, AraBERT nominally ahead on both HARD variants) is consistent with a small real advantage on tweets and reviews with dialect, but it is not shown.
- A claim that Arabic-specific pretraining beats multilingual in general. Our evidence is one XLM-R configuration (plus one fixed alternative) and the other two models.
- Comparing HARD numbers with published ones: our split is by hotel, our training set is a 20,000-review subsample, and our file has 105,698 rows (Chapters 3 and 4).

## Seeds and epoch selection

- **Seed spread** (standard deviation over 3 seeds): MARBERT 0.010 to 0.011, XLM-R 0.009 to 0.011, AraBERT 0.011 on AJGT and **0.032 on ASTD**. The AraBERT ASTD figure comes from one run: seed 2 picked epoch 1 on a validation peak and scored 0.8306, while seeds 0 and 1 scored 0.8786 and 0.8904.
- **Early peaks:** in 9 of the 18 stage-1 runs the best epoch was 1 or 2 of 5. They were all Arabic-specific models (AraBERT 5 of 6, MARBERT 4 of 6), none XLM-R, whose best epochs were 3, 4 or 5. Training losses at the end: AraBERT and MARBERT 0.02 to 0.10, XLM-R 0.23 to 0.31. So the Arabic models fit the training data quickly and then drift, while XLM-R is still learning when the schedule ends.
- **Seed matters more than model on AJGT.** The seed spread (about 1 point) is similar to the gap between AraBERT and MARBERT (1.7 points).
- Epoch selection on tiny validation sets is noisy (180 and 242 rows). Averaging the last few epochs, or choosing a fixed epoch count, would avoid the failure mode. Not tried.

## XLM-R: what the experiments show

First hypothesis: XLM-R is undertrained at lr 2e-5 on small data. Evidence for: its final training loss is 3 to 10 times higher than the other models', and its validation score was still rising at epoch 5 in 3 of 6 runs.

The test: one alternative configuration, fixed beforehand (lr 3e-5, 10 epochs, batch 16). Results: AJGT improved from 0.8685 to 0.9037 (+3.5), ASTD got worse from 0.8367 to 0.8067 (-3.0). On ASTD the validation score peaked at epoch 2 or 3 and then fell while training loss kept falling, which is overfitting.

What this means:
- "Undertrained" was half right. More training helps on AJGT and hurts on ASTD, and I predicted the wrong sign for ASTD.
- Three settings changed at once (learning rate, epochs, batch size), so I cannot say which one caused either result.
- Both rows are reported. The better one was not picked per dataset.

A pattern worth noting: the gap between XLM-R and MARBERT shrinks as training size grows.

| Train rows | Dataset | XLM-R minus MARBERT, in points |
|---|---|---|
| 1,440 | AJGT | -5.9 |
| 1,935 | ASTD | -5.3 |
| 8,301 | LABR | -3.2 |
| 20,000 | HARD, as released | -0.3 |
| 20,000 | HARD, descriptor removed | -0.4 |

This is consistent with XLM-R needing more data or updates, but datasets differ in domain and variety, and the larger ones are single-seed, so it is a hypothesis. A cheap test: train both models on LABR subsamples of 1,500 and 4,000 rows. Not run.

## What is not equal across systems

An interviewer can probe each of these.

1. **AraBERT got raw text,** while its authors recommend their preprocessing tool. Its unknown-piece rate on reviews was 1.8 to 2.1%, mostly content words (Chapter 8). Its numbers may be understated.
2. **HARD encoders trained on 20,000 rows,** TF-IDF on all 81,906. The comparison is therefore stricter for the encoders. Whether more data would change any ranking was not tested.
3. **Seeds differ by dataset:** 3 on ASTD and AJGT, 1 elsewhere.
4. **No hyperparameter search for any model,** one fixed XLM-R alternative. A tuned AraBERT or MARBERT might also move.
5. **The batch differs for XLM-R on reviews** (16 with accumulation 2, effective 32). The effective batch matches, but gradient noise per step does not.
6. **Max length** 96 for tweets and 256 for reviews. A MARBERT run at 128 tokens gave 0.8844 against 0.8799, which is no detectable difference (paired interval [-0.0048, +0.0137], Chapter 8).
7. **MARBERT's pretraining may include ASTD and AJGT tweets.** Not checked.
8. **Model choice used test scores.** MARBERT was picked for serving partly because it led on test. With three candidates and these gaps the risk is small.

## Paired comparisons (script provided, results pending)

Interval overlap is a rough rule. A paired bootstrap on the same test rows is more powerful, and `paired.py` already does it for two files. This script runs it for every dataset and every pair of systems and prints all of them. For ASTD and AJGT it reports seed 0 with an interval and the seed 1 and 2 differences as points.

```bash
nano src/arabic_sentiment/paired_all.py
```

```python
"""Paired bootstrap of macro-F1 differences between systems on identical test rows.
Run: uv run python -m arabic_sentiment.paired_all
Output: results/paired.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "results/preds"
DATASETS = ["labr", "hard", "hard_notitle", "astd", "ajgt"]
PAIRS = [("arabert", "tfidf"), ("marbert", "tfidf"), ("xlmr", "tfidf"),
         ("marbert", "arabert"), ("marbert", "xlmr"), ("arabert", "xlmr")]
N_BOOT, SEED = 2000, 42


def macro_f1(y, p):
    tp1, tp0 = np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 0))
    fp1, fn1 = np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p == 0))
    return (2 * tp1 / max(2 * tp1 + fp1 + fn1, 1) + 2 * tp0 / max(2 * tp0 + fp1 + fn1, 1)) / 2


def find(d, m, s):
    pat = re.compile(rf"{d}-{m}-s{s}(-n\d+)?\.parquet")
    hits = [f for f in PREDS.glob(f"{d}-{m}-s{s}*.parquet") if pat.fullmatch(f.name)]
    return hits[0] if hits else None


def main():
    rows = []
    for d in DATASETS:
        for a, b in PAIRS:
            res = {}
            for s in (0, 1, 2):
                fa, fb = find(d, a, s), find(d, b, 0 if b == "tfidf" else s)
                if fa is None or fb is None:
                    continue
                A = pd.read_parquet(fa)[["text", "label", "pred"]]
                B = pd.read_parquet(fb)[["text", "pred"]]
                m = A.merge(B, on="text", suffixes=("_a", "_b"))
                assert len(m) == len(A) == len(B), f"{d} {a} vs {b} seed {s}: row mismatch"
                y, pa, pb = m.label.to_numpy(), m.pred_a.to_numpy(), m.pred_b.to_numpy()
                res[s] = (macro_f1(y, pa) - macro_f1(y, pb), y, pa, pb)
            if 0 not in res:
                continue
            diff0, y, pa, pb = res[0]
            rng = np.random.default_rng(SEED)
            boots = np.empty(N_BOOT)
            for k in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                boots[k] = macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"dataset": d, "pair": f"{a} - {b}", "n": len(y), "diff_s0": round(diff0, 4),
                         "ci95": f"[{lo:+.4f}, {hi:+.4f}]", "excludes_0": bool(lo > 0 or hi < 0),
                         "diff_s1": round(res[1][0], 4) if 1 in res else None,
                         "diff_s2": round(res[2][0], 4) if 2 in res else None})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/paired.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
```

```bash
uv run python -m py_compile src/arabic_sentiment/paired_all.py && uv run python -m arabic_sentiment.paired_all
```

It runs on CPU in a minute or two. Its pairing uses each system's seed-0 predictions (and seed 1 and 2 where they exist, for the 3-seed datasets). Differences are in macro-F1, as fractions (0.05 = 5 points).

Predictions committed before running, so they can be wrong:
- MARBERT against TF-IDF: the interval excludes 0 on LABR, HARD without the descriptor and ASTD. AJGT is a coin flip.
- AraBERT against TF-IDF: excludes 0 on LABR, both HARD variants and ASTD. It includes 0 on AJGT.
- XLM-R against TF-IDF: excludes 0 on LABR and both HARD variants. It includes 0 on AJGT, and ASTD is unclear.
- MARBERT against AraBERT: includes 0 on HARD, ASTD and AJGT. LABR is unclear.

## What went wrong

| Item | Detail |
|---|---|
| Wrong sign for XLM-R on ASTD | I predicted the tuned configuration would help. It hurt (0.8367 to 0.8067) |
| Noisy epoch choice | AraBERT ASTD seed 2 (see above) |
| README overstated | "Beat TF-IDF on every dataset" and "best or tied-best on 4 of 5" (see the patch below) |
| Duplicate subsample patch, name collisions, ablation run name | Chapter 9 |
| Partial runs read as results | An early comparison printed HARD with n=1 (Chapter 6) |

## Not done

- A hyperparameter search for any model.
- More seeds on LABR and HARD, or on the LLM runs.
- The AraBERT preprocessing experiment, and the model-size learning curve above.
- Calibration (do the confidence scores mean anything?). The deployed API returns a confidence, so this matters.
- Error analysis by length, by class, or by hotel. Chapter 14 covers errors on three datasets only.

## Check yourself

1. Which datasets show a gain of an encoder over TF-IDF that is clearly outside noise, and which one does not?
2. Why can MARBERT and AraBERT not be ranked from the main table, and what would you run to rank them?
3. Why is the AraBERT ASTD standard deviation 0.032, and what does it say about epoch selection?
4. The tuned XLM-R helped on AJGT and hurt on ASTD. Why can you not say which setting caused it?
5. How does the XLM-R gap change with training size, and why is that only a hypothesis?
6. List three ways the encoder comparison is not equal across systems.
7. Why is HARD without the descriptor the better HARD number, and why can it not be compared with published HARD results?
