---
language: ar
license: other
base_model: UBC-NLP/MARBERT
pipeline_tag: text-classification
tags: [arabic, sentiment-analysis, dialect]
---

# Arabic sentiment (binary), MARBERT fine-tuned on four datasets

Fine-tuned from UBC-NLP/MARBERT to label Arabic text as negative or positive (MSA and dialect, reviews and tweets).
Code, benchmark, error analysis and a study guide: https://github.com/Muhamm-dk/arabic-sentiment-benchmark

## Labels
`negative` (0) and `positive` (1). There is no neutral class.

## Training data
31,674 training rows: HARD hotel reviews (a 20,000-row subsample, with the leading Booking.com score phrase removed by exact match), LABR book reviews (8,301), ASTD Egyptian tweets (1,935, Negative and Positive only) and AJGT Jordanian tweets (1,438). Star ratings 1 and 2 are negative and 4 and 5 are positive. Duplicates and texts with conflicting labels were removed before splitting.

## Training procedure
Full fine-tuning with AdamW (learning rate 2e-5, weight decay 0.01), a linear schedule with 10% warmup, batch size 32, 3 epochs, bf16 autocast, maximum length 256 tokens, seed 0. The epoch with the best validation macro-F1 was kept (epoch 2, 0.8984).

## Test results (macro-F1, held-out test splits, one seed, 95% bootstrap interval)
| Dataset | Test rows | Macro-F1 |
|---|---|---|
| AJGT | 180 | 0.933 [0.898, 0.967] |
| ASTD | 242 | 0.869 [0.823, 0.914] |
| HARD (descriptor removed) | 10,707 | 0.908 [0.902, 0.913] |
| LABR | 2,283 | 0.864 [0.849, 0.877] |

Against MARBERT models trained on a single dataset: not detectably different on AJGT, ASTD and HARD, and 1.6 points lower on LABR (paired interval [-2.8, -0.4]).

## Using it
Texts longer than 256 tokens are truncated. The score returned by the service is the larger of the two softmax probabilities.

## Limitations
- Binary only: no neutral class. Neutral or factual text is forced into a polarity.
- The confidence score is overconfident. On LABR, predictions with confidence between 0.7 and 0.99 were right 65% to 88% of the time, and those at 0.99 or more about 98% of the time. Use it to rank predictions, not as a probability of being right.
- Small test sets (AJGT 180, ASTD 242): differences of a few points are within noise.
- Political tweets: labels often follow the author's stance, not the wording, and the model misreads these.
- Mixed reviews ("good but...") are a frequent error source.
- Removing the HARD score phrase also stripped quoted user titles in about 6% of HARD rows.
- Dialect labels were not used in training. A per-dialect check with a third-party classifier found no detectable gap on LABR and HARD, with the caveats in the repository. Gulf and Maghrebi text was not evaluated.
- Not tested: other domains, very short messages, text mixed with English or emoji.
- The four training datasets do not state licenses. Use this model for research, and check the original dataset sources before any commercial use.
