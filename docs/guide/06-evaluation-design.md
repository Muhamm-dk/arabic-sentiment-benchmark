# Chapter 6: Evaluation design

## What

The rules that decide how every system is scored and compared: which metric, how uncertainty is measured, how many seeds, which split selects and which judges, and which comparisons are allowed.

## Why: the decisions

### 1. Macro-F1 as the headline metric

F1 for one class is the harmonic mean of precision and recall. Macro-F1 computes F1 for each class and averages them with equal weight.

Why:
- ASTD is 68% negative once binarized, and LABR's full set is skewed toward positive. A model that predicts the majority class gets high accuracy and a poor macro-F1.
- One number per system, comparable across datasets.

Accuracy is reported next to it in the result files. On the balanced test sets (AJGT, LABR official, HARD) the two are almost equal (TF-IDF on HARD: accuracy 0.9411, macro-F1 0.9411), so the choice only matters on ASTD. I did not check which metric the published papers use.

### 2. Bootstrap confidence intervals on the test set

```python
def bootstrap_ci(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(f1_score(y[i], p[i], average="macro"))
    return [round(float(x), 4) for x in np.percentile(scores, [2.5, 97.5])]
```

- `rng.integers(0, len(y), len(y))` draws `len(y)` row indices with replacement, so a resample is a plausible alternative test set of the same size.
- The model is not retrained. Only its fixed predictions are resampled.
- The 2.5th and 97.5th percentiles of 1,000 resampled scores form a 95% interval (the percentile method).
- The seed makes the interval reproducible.

What it measures: uncertainty from which test rows happened to be drawn. It does not measure training noise (different seeds) or noise from the choice of split.

Why it matters here: ASTD has 242 test tweets and AJGT has 180. Scores that look 4 points apart are inside noise there.

### 3. Seeds

- ASTD and AJGT: 3 seeds per encoder (0, 1, 2), reported as mean and standard deviation (pandas, sample standard deviation).
- LABR and HARD: one seed. Their test sets are large (2,283 and 10,707), so interval widths are about 3 and 1 points, but training variance was not measured on them.
- LLMs: temperature 0, one run. This removes sampling randomness, but prompt variation and hosted-model changes are not measured.

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

The seed sets Python, NumPy and PyTorch random states, which controls classifier head initialization and dropout. Batch order uses a separate generator, `np.random.default_rng(seed)`. Bit-for-bit reproducibility on the GPU under bf16 was not tested.

### 4. Selection on validation, judgement on test

```python
for epoch in range(1, a.epochs + 1):
    ...train one epoch...
    val_f1 = macro_f1(y["val"], predict_proba(model, enc["val"], y["val"], tok))
    if val_f1 > best_f1:
        best_f1, best_epoch = val_f1, epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
probs = predict_proba(model, enc["test"], y["test"], tok)
```

- Each epoch is scored on validation. The weights of the best epoch are copied to CPU memory.
- After training, those weights are restored and test is scored exactly once. The test split is encoded at the start but is not used by anything until the last lines, so it cannot influence the choice.
- `>` is strict, so a tie keeps the earlier epoch.
- The same rule applies to TF-IDF (configuration chosen on validation), the LLM prompt (language chosen on 60 validation tweets), and everything else.

### 5. Rules fixed before looking

- English prompt over Arabic only if Arabic wins by 5 or more points on validation.
- XLM-R "tuned" configuration (lr 3e-5, 10 epochs) fixed once, and both XLM-R rows reported whatever happened.
- Invalid LLM replies count as errors.
- Test sizes for LLMs (300 for LABR and HARD) set from quotas before running.

### 6. Same rows for every comparison

Every comparison table uses identical test rows for all systems, which Chapter 12 covers. The 300-row LLM comparison cannot be placed next to full-test numbers: TF-IDF scores 0.883 on the 300 LABR rows and 0.8155 on all 2,283.

### 7. How LLM outputs are scored

```python
def macro_f1(y, p):
    return float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0))
```

An unparseable reply is stored as -1. With `labels=[0, 1]` the score is averaged over the two real classes only, so a -1 never becomes a third class, but it is a wrong prediction for its row (it lowers recall for the true class). The invalid rate is reported next to each score.

## Alternatives

| Choice | Alternative | Verdict |
|---|---|---|
| Macro-F1 | Accuracy | Misleading on ASTD. Kept as a secondary figure |
| | Weighted F1, MCC, ROC-AUC | Not used. MCC is a reasonable single number for imbalance. AUC needs scores and was not computed for the LLMs |
| | Calibration (Brier score, ECE) | Not measured. Matters for a deployed classifier, since the API returns a confidence |
| Bootstrap CI on predictions | k-fold cross-validation | Better for tiny datasets, but every run is multiplied, which was impossible for the LLMs |
| | Repeated random splits | Same cost problem |
| Overlap of intervals | Paired bootstrap on identical rows | More powerful, because the rows are identical. Run later for encoders and TF-IDF (Chapter 10, `paired_all.py`) and for LLMs (Chapter 12, `paired_llm.py`). It measures test-row sampling only, not training-seed variation |
| | McNemar's test | Same: possible, not run |
| | Wilcoxon over seeds | Needs more than 3 seeds |
| 3 seeds on small sets | 5 or 10 seeds | More reliable, more compute |
| Single seed on LABR and HARD | 3 seeds | Training variance unmeasured there. HARD runs also use a 20,000-review subsample |
| Test used once per run | Nested cross-validation | Unnecessary at this scale |

## What went wrong

| Item | Detail |
|---|---|
| Noisy validation picks the wrong epoch | AraBERT on ASTD, seed 2: best validation macro-F1 was 0.8770 at epoch 1, test was 0.8306. Seeds 0 and 1 scored 0.8786 and 0.8904 on test. A 242-tweet validation set cannot separate epochs a few points apart. That one run is why AraBERT's ASTD standard deviation is 0.032 |
| Partial results looked like scores | An early `compare.py` printed HARD with n=1 and F1 1.0 because a queue had stopped after one row. The fix was to trust a score only when its summary file exists |
| Mixing row sets | The 300-row LABR sample is easier than the full test set, so TF-IDF moved from 0.8155 to 0.883 |
| An explanation that was half wrong | I predicted that tuning XLM-R would fix it on both small sets. It helped AJGT (0.8685 to 0.9037) and hurt ASTD (0.8367 to 0.8067). Lesson: report both rows and do not pick the better one |
| Seed asymmetry | AJGT and ASTD encoder scores are 3-seed means, LLM scores single runs, LABR and HARD encoder scores single seeds. The README states this, and comparisons should not hide it |

## Where the protocol was not followed to the letter

These do not change any conclusion, but they are the places an interviewer could probe:

1. **Test rows were used to choose the serving model family.** MARBERT was picked partly because it scored highest on test in 4 of 5 comparisons. With gaps of several points and three candidates, the risk is small, but it is a selection on test.
2. **The deployed model was kept after seeing per-dataset test scores** (combined model against single-dataset models). The decision rule was "within noise", written before the comparison, but the numbers were visible.
3. **Diagnostics ran on test splits.** The HARD strict-rule check scored TF-IDF on the test set. It was used for documentation, not to choose a configuration.
4. **The AJGT smoke test** (one MARBERT run) was scored on test, and the result is a reported number.
5. **Significance tests came late.** Chapters 6 to 9 were written using interval overlap, a rough rule. Paired bootstrap tests were added afterwards (Chapter 10) and changed two statements: MARBERT over AraBERT on LABR is supported, and the encoders are slightly separable on HARD. The tests cover test-row sampling only, not training-seed variation, and 30 uncorrected intervals were examined.
6. **The gpt-oss output budget** was changed once on validation after seeing the invalid rate (300 to 1000 tokens), then frozen.

## Numbers

Interval widths from our runs show what each test size supports:

| Test set | n | Example interval | Width |
|---|---|---|---|
| AJGT | 180 | TF-IDF [0.8217, 0.9166] | 9.5 points |
| ASTD | 242 | TF-IDF [0.7095, 0.8277] | 11.8 |
| LABR | 2,283 | TF-IDF [0.7986, 0.8304] | 3.2 |
| HARD no-descriptor | 10,707 | TF-IDF [0.8648, 0.8772] | 1.2 |
| HARD | 10,707 | TF-IDF [0.9366, 0.9454] | 0.9 |
| 300-row LLM sample | 300 | Qwen on HARD no-descriptor [0.8221, 0.9022] | 8.0 |

Seed noise (standard deviation of test macro-F1 over 3 seeds): AraBERT ASTD 0.032, MARBERT ASTD 0.010, AraBERT AJGT 0.011, MARBERT AJGT 0.011, XLM-R 0.009 to 0.011.

## Rules for reading any table in this guide

1. A gap smaller than about half the interval width is noise.
2. On ASTD and AJGT, only gaps of 5 points or more count as likely real. This is my rule of thumb, not a derived threshold.
3. Single-seed results on large test sets are fine for big gaps (3 points or more on LABR) and not for small ones.
4. 300-row numbers are never compared with full-test numbers.
5. Always look at the invalid rate next to an LLM score.

## Check yourself

1. Why does macro-F1 punish a model that predicts only the majority class?
2. What does a bootstrap interval measure, and what does it miss?
3. Where in `finetune.py` is the guarantee that test does not influence epoch selection?
4. AraBERT on ASTD seed 2 picked epoch 1 and scored 0.8306. What went wrong, and what would you change?
5. Two intervals overlap. Can you say the systems are equal? What would you run instead?
6. Why can the 300-row LABR TF-IDF score not be compared with 0.8155?
7. Name two places where test data was used for something other than a final score, and why they do not change the conclusions.
8. Why is an LLM's invalid reply counted as wrong, and how does `labels=[0, 1]` keep it from becoming a third class?
