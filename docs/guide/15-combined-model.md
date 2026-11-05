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
- Against the single-dataset MARBERT, the differences are within the interval widths on AJGT, ASTD and HARD. The LABR difference is borderline.
- The comparison is not equal: the combined model is one seed, and the single-model figures for AJGT and ASTD are 3-seed means.

Paired comparison and calibration (`combined_checks.py`): (fill in after running).

## What went wrong

| Item | Detail |
|---|---|
| **Wrong explanation of 1,438** | I called 1,438 the "post-dedupe" AJGT figure. `data.py` printed 1,800 rows before and after dedupe. The real cause: two AJGT train tweets ("لا تعليق" and "جميل جدا") also occur in LABR train, and `drop_duplicates` kept the LABR copies. Their labels were not compared (check below) |
| No cross-dataset label check | The first copy wins whatever its label |
| Pooled score is misleading | 79.8% of test rows are HARD |
| First reading too soft | I first said the combined model was roughly on par with the single-dataset models, from interval widths. Paired intervals replace that |
| Selection after seeing test results | The combined model was kept after per-dataset test scores were visible. The rule was written beforehand ("within noise"), but the numbers were visible |
| Validation dominated by HARD | Epoch selection follows HARD and not the small datasets |

Label check for the two tweets: (fill in after running). Expectation, not a measurement: "جميل جدا" is positive in both datasets, and "لا تعليق" may differ.

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
