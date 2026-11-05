# Chapter 15: The combined model

## What

One MARBERT fine-tuned on all four datasets at once, the model the API serves. Code: `combined.py` (builds the training set), `finetune.py --dataset combined` (Chapter 9), `combined_report.py` (per-dataset scores), `combined_checks.py` (paired comparison and calibration).

## Why one model, and why MARBERT

**One model.** A deployed service receives any kind of text: MSA and dialect, reviews and tweets. Routing to four specialists would need a dataset or dialect classifier, and Chapter 13 shows dialect classification is unreliable on short text. One model is one artifact (one model card, one image, 650 MB on the Hub). The risk is that mixing domains costs accuracy on some of them. The experiment measures that. Pooling might also help the two tiny datasets.

**MARBERT.** Top score on LABR, ASTD and AJGT, and within 0.3 points of AraBERT on both HARD variants (Chapter 10). The paired tests support MARBERT over AraBERT on LABR only (+1.7), so the choice between those two is weakly supported. The model was chosen partly from test scores (Chapter 6). It has 163M parameters against AraBERT's 135M, so it is slightly heavier. Measured in the container: 405 MiB of memory after loading and one request (Chapter 17). CPU latency of the two was not compared. XLM-R was rejected on accuracy for tweets and size (278M parameters).

## Training set design

| Dataset | Train rows | Share of training data |
|---|---|---|
| HARD without the descriptor (capped) | 20,000 | 63.1% |
| LABR | 8,301 | 26.2% |
| ASTD | 1,935 | 6.1% |
| AJGT | 1,438 | 4.5% |
| Total | 31,674 | |

Validation holds 11,766 rows (HARD 88.6%) and test 13,412 (HARD 79.8%).

1. **HARD without the descriptor, not raw HARD.** A user does not type a Booking.com score label, so training on `hard` would teach a shortcut the API never sees. Alternative: both variants as augmentation, which could make the model robust to either. Not tried.
2. **HARD capped at 20,000 training rows,** the same size as the Stage 3 runs. Without the cap, HARD would be 87.5% of the data (81,906 of 93,580 rows). The cap uses the same seed and positional sampling as the single runs, so it should be the same 20,000 rows. Not verified.
3. **Even capped, HARD is the largest part.** Alternatives: upsample the small sets, sample datasets in proportion to a power of their size, weight the loss. None were tried.
4. **Original splits kept.** Validation and test rows come from the same splits as the benchmark, so each dataset's test set is identical to the one used for every other system.
5. **Same recipe as the single runs:** MARBERT, lr 2e-5, batch 32, 3 epochs, max length 256, seed 0, best epoch on validation.
6. **One pooled validation set picks the epoch,** so HARD decides it. The three validation scores differ by 0.3 points (0.8950, 0.8984, 0.8972), so epoch 2 over epoch 3 is a near tie.

## How: `combined.py`

```python
PARTS = ["labr", "hard_notitle", "astd", "ajgt"]
HARD_TRAIN_CAP = 20_000

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
out = out.drop(leak.index)
```

- HARD alone is subsampled, and only its train rows. Validation and test rows are kept whole.
- `concat` stacks the four datasets in the order above, keeping each row's `dataset` column.
- `drop_duplicates(["text", "split"])` removes a text that appears twice in the same split across datasets and **keeps the first one in concatenation order** (LABR first). It does not check whether the two copies carry the same label. If they disagree, the earlier dataset's label wins silently.
- The leak check drops validation or test rows whose text also occurs in train. It printed 0.
- The check at the end of the script prints the counts per split and dataset, and the positive share per split (train 0.485, validation 0.537, test 0.492).

Training: `uv run python -m arabic_sentiment.finetune --dataset combined --model marbert --seed 0 --epochs 3 --save`. The run is named `combined-marbert-s0`, the best-epoch weights are saved to `outputs/combined-marbert-s0` (not in git), and its test predictions carry a `dataset` column so they can be scored per dataset.

## Results

Training log: train loss 0.3559, 0.2052, 0.1187 and validation macro-F1 0.8950, 0.8984, 0.8972 for epochs 1 to 3. Best epoch 2. Pooled test macro-F1 0.9004, interval [0.8952, 0.9055], n 13,412. Peak VRAM 5.18 GiB.

The pooled score is dominated by HARD (79.8% of test rows) and says little. Per dataset:

| Dataset (test n) | Combined | 95% interval | Single-dataset MARBERT | Difference | TF-IDF |
|---|---|---|---|---|---|
| AJGT (180) | 0.9333 | [0.898, 0.967] | 0.9277 (3-seed mean) | +0.6 | 0.8722 |
| ASTD (242) | 0.8691 | [0.823, 0.914] | 0.8902 (3-seed mean) | -2.1 | 0.7708 |
| HARD, descriptor removed (10,707) | 0.9078 | [0.902, 0.913] | 0.9045 (one seed) | +0.3 | 0.8710 |
| LABR (2,283) | 0.8641 | [0.849, 0.877] | 0.8799 (one seed) | -1.6 | 0.8155 |

- The combined model beats TF-IDF everywhere (+6.1, +9.8, +3.7, +4.9 points).
- Against the single-dataset MARBERT, the paired intervals include 0 on AJGT, ASTD and HARD. The LABR difference (-1.6) is measurable (interval [-2.8, -0.4]).
- The comparison is not equal: the combined model is one seed, and the single-model figures for AJGT and ASTD are 3-seed means.

### Paired comparison with the single-dataset models (`combined_checks.py`)

Combined minus single-dataset MARBERT, in points. The interval is a 95% paired bootstrap for seed 0, the only seed on LABR and HARD:

| Dataset (n) | Seed 0 | Interval | Seeds 1 and 2 |
|---|---|---|---|
| LABR (2,283) | -1.6 | [-2.8, -0.4] | one seed only |
| HARD, descriptor removed (10,707) | +0.3 | [-0.1, +0.7] | one seed only |
| ASTD (242) | -3.0 | [-6.7, +0.5] | -2.3, -1.1 |
| AJGT (180) | +1.7 | [-1.1, +4.6] | -0.6, +0.6 |

- **LABR:** the combined model is lower and the interval excludes 0. One seed each, and test-row sampling only. This is the one measurable cost of pooling. I earlier called it borderline, and the paired interval says it is supported.
- **ASTD:** lower in all three comparisons, but the seed-0 interval includes 0. The mean of the three single-dataset seeds is 2.1 points above the combined score.
- **AJGT and HARD:** no detectable difference. The AJGT differences change sign across seeds.
- Predictions made before running, scored: the LABR interval excludes 0 (held), the other three include 0 with HARD narrowly (held), and calibration (half held, below).

### Calibration of the deployed model's confidence

Same method as Chapter 14 (confidence is the larger softmax probability). Cells give accuracy, mean confidence in brackets, and the row count.

| Dataset | [0.5, 0.7) | [0.7, 0.9) | [0.9, 0.99) | [0.99, 1] | Expected calibration error | At 0.9 or more: accuracy against confidence |
|---|---|---|---|---|---|---|
| LABR (2,283) | 0.546 (0.600), 152 | 0.647 (0.822), 255 | 0.875 (0.972), 1,106 | 0.983 (0.995), 770 | 0.074 | 0.920 against 0.981 |
| HARD, no descriptor (10,707) | 0.553 (0.604), 514 | 0.697 (0.816), 858 | 0.917 (0.976), 5,454 | 0.988 (0.996), 3,881 | 0.045 | 0.947 against 0.984 |
| ASTD (242) | 0.556 (0.605), 9 | 0.690 (0.824), 29 | 0.890 (0.969), 109 | 0.979 (0.995), 95 | 0.060 | 0.931 against 0.981 |
| AJGT (180) | 0.750 (0.621), 8 | 0.667 (0.808), 18 | 0.969 (0.970), 96 | 0.983 (0.995), 58 | 0.025 | 0.974 against 0.980 |

- **Against the single-dataset MARBERT models (Chapter 14):** the expected calibration error is lower for the combined model on LABR (0.074 against 0.102), HARD (0.045 against 0.066) and AJGT (0.025 against 0.075), and higher on ASTD (0.060 against 0.028). The share of rows with confidence 0.99 or more is 33.7% against 85.4% on LABR and 36.2% against 50.9% on HARD. A guess, not tested: the combined model was selected at epoch 2 (training loss 0.205), while the single LABR run was selected at epoch 4 (0.069).
- **Still overconfident in the middle.** On LABR, rows with confidence 0.7 to 0.99 are right 65% to 88% of the time while the model reports 82% to 97%.
- **Rows at 0.99 or more are right 97.9% to 98.8% of the time** on the four test sets. Rows below 0.9 are right far less often: LABR 407 rows (17.8%) at about 61%, HARD 1,372 rows (12.8%) at about 64% (my arithmetic from the bins).
- **A possible API rule:** mark confidence below 0.9 as uncertain. That threshold was read from test-set tables, so it should be set on validation predictions before shipping. Not done.
- **The bins are small on the small sets:** AJGT's two lowest bins hold 8 and 18 rows, ASTD's 9 and 29.
- Prediction 3, scored: the gap at 0.9 or more exceeds 3 points on LABR, ASTD and HARD but is 0.6 on AJGT; it is largest on LABR; LABR's expected calibration error is 0.074.

## What went wrong

| Item | Detail |
|---|---|
| **Wrong explanation of 1,438** | I called 1,438 the "post-dedupe" AJGT figure. `data.py` printed 1,800 rows before and after dedupe. The real cause: two AJGT train tweets ("لا تعليق" and "جميل جدا") also occur in LABR train, and `drop_duplicates` kept the LABR copies. Their labels agree (checked below) |
| No cross-dataset label check | The first copy wins whatever its label |
| Pooled score is misleading | 79.8% of test rows are HARD |
| First reading too soft | I first said the combined model was roughly on par with the single-dataset models, from interval widths. Paired intervals replace that |
| Selection after seeing test results | The combined model was kept after per-dataset test scores were visible. The rule was written beforehand ("within noise"), but the numbers were visible |
| Validation dominated by HARD | Epoch selection follows HARD and not the small datasets |

Label check for the two dropped AJGT tweets: they carry the same labels in both datasets ("لا تعليق" negative and "جميل جدا" positive, in AJGT train and in LABR train). Dropping the AJGT copies lost no information. My expectation that "لا تعليق" might differ was wrong.

## Not done

- More seeds, other mixing ratios, upsampling, a loss weighted by dataset.
- Training on both HARD variants as augmentation.
- A neutral class.
- Temperature scaling or other calibration.
- A dialect breakdown of the combined model.
- Out-of-domain tests: Gulf tweets, product or app reviews, very short messages, text mixed with English or emoji. The API will meet such text, and no result here covers it.

## Check yourself

1. Why is the pooled test score 0.9004 not a useful headline?
2. Why `hard_notitle` and not `hard` in the combined training set?
3. What does `drop_duplicates(["text", "split"])` keep, and what check is missing?
4. What caused AJGT's 1,438 against 1,440 train rows, and what was wrong with my first explanation?
5. Why does HARD decide the epoch choice, and how large is the difference between the epochs?
6. Why is the single-versus-combined comparison not equal?
7. What kind of text would the deployed model receive that none of the four datasets covers?
