# Chapter 14: Error analysis

## What

Where and why MARBERT goes wrong. Two views: a manual read of 61 errors tagged by cause, and counts from the saved predictions (`error_stats.py`). Chapter 13 covers errors by dialect group, and this chapter covers causes.

## Who read what, and the limits

- **Annotator: an AI assistant (Claude), not a native-speaker human.** Its reading of dialect and of political context is fallible. Dialect-vocabulary calls in particular need a native reader.
- **Sample:** MARBERT, seed 0, test errors. AJGT: all 15 of 180. ASTD: all 21 of 242. LABR: 25 drawn at random (seed 0) from 274 errors, so 9% of them. HARD was not read.
- **One annotator, categories invented while reading,** not fixed in advance. Categories overlap, and each error went into its most likely one.
- AraBERT, XLM-R and the LLMs were not analysed.

## The categories and counts

| Category | Meaning | AJGT (15) | ASTD (21) | LABR (25) | Total (61) |
|---|---|---|---|---|---|
| No clear sentiment | neutral, factual, a headline, hashtags, a name or a fragment carrying a polarity label | 6 | 9 | 3 | 18 (30%) |
| Mixed or hedged | praise and criticism together | 0 | 2 | 11 | 13 (21%) |
| Gold label arguably wrong | a reader would give the other label | 2 | 3 | 4 | 9 (15%) |
| Needs outside context | stance, sarcasm, politics, religion, world knowledge | 4 | 6 | 3 | 13 (21%) |
| Dialect vocabulary | plausibly caused by dialect words or idioms | 3 | 1 | 4 | 8 (13%) |

LABR figures come from 25 of 274 errors. 11 of 25 is 44%, with an uncertainty of roughly plus or minus 20 points.

### Examples (short, for illustration)

- **No clear sentiment:** AJGT "صراع متصاعد للهيمنه على سوق الالمنيوم العالميه" (a neutral news line, gold negative), ASTD "نصر محروس" (a bare name, gold negative).
- **Mixed:** LABR "رواية ممتازة فى اسلوبها ولكنها تدعو إلى الفجور" (gold negative, predicted positive).
- **Needs context:** ASTD "الانقلاب يريد قمع 50 مليون مصرى..." (gold positive, predicted negative). The label follows the author's side, while the wording reads as grievance.
- **Gold label arguably wrong:** ASTD "شرييييفة! #فيفي_عبده" (gold negative).
- **Dialect vocabulary:** AJGT "صار بدها بطاريه التقوى شحن" and "عيش زي ما بدك ما ترد على حدا ابدا" (both gold positive, predicted negative).

## What the read suggests

1. **Label quality and missing sentiment are the largest buckets:** no clear sentiment plus gold arguably wrong is 27 of 61 (44%). Many errors are the benchmark's, not the model's. That is also why scores near 0.9 are hard to push higher.
2. **Mixed reviews dominate LABR's sample** (11 of 25). A star rating summarizes praise and criticism into one label.
3. **ASTD labels often follow stance,** not wording (6 of 21 needed context).
4. **Dialect vocabulary explains 8 of 61 (13%),** AJGT 3 of 15, LABR 4 of 25 and ASTD 1 of 21. With one AI annotator and a LABR sample of 25, I would not defend any figure more precise than "a minority".
5. **Direction of the errors** (from the printed lists, my count; `error_stats.py` verifies): AJGT 9 false positives and 6 false negatives, ASTD 9 and 12, the LABR sample 11 and 14. On ASTD positive is the minority class (32%), so about 12 of roughly 77 positive tweets were missed (about 16%), against about 9 of 165 negative tweets wrongly flagged (about 5%). Plain cross-entropy was used with no class weights (Chapter 9), which is a plausible contributor but was not tested.

### Predicted dialect of the errors

From the printed lists, using any-confidence tags and the test-set counts of Chapter 13:

| Dataset | Group | Errors | Rows | Error rate |
|---|---|---|---|---|
| ASTD | EGY | 10 | 107 | 9.3% |
| ASTD | MSA | 10 | 116 | 8.6% |
| AJGT | MSA | 6 | 36 | 17% |
| AJGT | LEV | 5 | 60 | 8% |
| AJGT | EGY | 2 | 20 | 10% |
| AJGT | GLF | 2 | 47 | 4% |

On ASTD the rates are similar (overall 8.7%). On AJGT, MSA-tagged tweets have about twice the overall error rate (8.3%), but from 6 errors, so I read it as too small to interpret. An earlier statement of mine said error rates were "similar across MSA, EGY and LEV rows". That is true on ASTD, but I had not computed rates for AJGT, and there it does not hold. The dialect classifier is also unreliable on short tweets (Chapter 13).

## Quantitative statistics (`error_stats.py`, seed 0)

Run `uv run python -m arabic_sentiment.error_stats`. It prints, for AJGT, ASTD, LABR and HARD without the descriptor: error rates of the four systems, MARBERT's false positives and false negatives, the rows all four systems get wrong, MARBERT's confidence on right and wrong answers, error rate by length quartile and by predicted dialect, and the most confident rows that all four systems get wrong (a pointer to label errors).

Results (`error_stats.py`, MARBERT seed 0, test sets; the AJGT and ASTD false-positive and false-negative counts match my hand counts):

| Dataset (n) | Error rate: TF-IDF / AraBERT / MARBERT / XLM-R | MARBERT errors (false positives / false negatives) | Wrong for all four | MARBERT errors shared with at least two others | Mean confidence right / wrong | Errors at confidence 0.9 or more |
|---|---|---|---|---|---|---|
| AJGT (180) | 12.8% / 8.9% / 8.3% / 12.2% | 15 (9 / 6) | 4 | 8 of 15 (53%) | 0.995 / 0.946 | 12 of 15 (80%) |
| ASTD (242) | 18.2% / 10.7% / 8.7% / 14.9% | 21 (9 / 12) | 7 | 13 of 21 (62%) | 0.945 / 0.861 | 12 of 21 (57%) |
| LABR (2,283) | 18.4% / 13.7% / 12.0% / 15.2% | 274 (150 / 124) | 93 | 177 of 274 (65%) | 0.987 / 0.943 | 226 of 274 (82%) |
| HARD no descriptor (10,707) | 12.9% / 9.3% / 9.6% / 9.9% | 1,023 (567 / 456) | 494 | 715 of 1,023 (70%) | 0.978 / 0.896 | 698 of 1,023 (68%) |

Error rate by length quartile (words):

- AJGT (up to 4, 5 to 7, 8 to 10, 11 or more; 53, 43, 44, 40 rows): 3.8%, 18.6%, 9.1%, 2.5% (2, 8, 4 and 1 errors, too few to interpret).
- ASTD (up to 10, 11 to 15, 16 to 20, 21 or more; 66, 60, 66, 50 rows): 7.6%, 11.7%, 7.6%, 8.0%.
- LABR (up to 13, 14 to 30, 31 to 64, 65 or more; 616, 533, 574, 560 rows): 13.5%, 10.7%, 9.8%, 13.9%.
- HARD without the descriptor (up to 9, 10 to 16, 17 to 29, 30 or more; 3,027, 2,512, 2,539, 2,629 rows): 13.0%, 8.7%, 9.1%, 6.8%.

Error rate by predicted dialect (any confidence): LABR EGY 9.6% (488 rows), GLF 12.1% (231), LEV 15.0% (153), MAGHREB 12.8% (257), MSA 12.4% (1,154). HARD without the descriptor: EGY 10.0% (80), GLF 8.5% (3,045), LEV 9.2% (840), MAGHREB 11.5% (419), MSA 10.0% (6,323). ASTD and AJGT are in the table above. LEV against EGY on LABR is a 5.4-point gap, about 1.7 standard errors by my arithmetic. Gulf-tagged HARD rows err 1.5 points less than MSA-tagged ones, about 2.4 standard errors treating rows as independent (they are not: reviews share hotels), and the Gulf label tracks hotel region (Chapter 13). I do not read either as a dialect effect.

Predictions made before running, scored:
1. **Held.** MARBERT errors shared with at least two other systems: 53%, 62%, 65%, 70%.
2. **Held.** Errors are less confident than correct answers on every dataset, and 57% to 82% of errors have confidence 0.9 or more.
3. **Failed.** Error rate falls with length on HARD (13.0% to 6.8%), is U-shaped on LABR, flat on ASTD, and not flat on AJGT (2, 8, 4 and 1 errors). On HARD the difference between the shortest and longest quartile is about 7.8 standard errors by my arithmetic. My guess, not tested: after the descriptor is stripped, short reviews have little text left.

What the statistics add to the manual read:
- A quarter to a half of MARBERT's errors are rows all four systems miss (27%, 33%, 34%, 48%). That is consistent with label noise and ambiguity, though hard rows would show the same pattern.
- Highest-confidence rows that all four systems miss (27 rows printed: 4 AJGT, 7 ASTD, 8 LABR, 8 HARD): 24 are gold-positive (AJGT 2 of 4, ASTD 6 of 7, LABR 8 of 8, HARD 8 of 8), against a base rate of about 50% (32% on ASTD). My reading of the visible text (cut at about 110 characters): all 8 HARD rows read as complaints under a positive label, 7 of 8 LABR rows show a criticism or contrast, and the ASTD rows are mostly political tweets whose label follows the author's side. Hypothesis: positive labels hide criticism more than negative labels hide praise. The star-rating check below does not support it as a general pattern: it describes only this high-confidence subset.
- Confidence is a weak error flag, which matters because the deployed API returns it (calibration below).

Not verified: whether HARD's review text holds a "liked" part and a "disliked" part (several complaint rows begin with a "none" phrase), and whether these rows reflect star ratings that are mildly positive.

### Error rate by original star rating

`error_extra.py`, seed 0, test sets. Error rate by the original star rating (3 stars were removed in Chapter 4):

| Dataset | Stars | Rows | TF-IDF | AraBERT | MARBERT | XLM-R |
|---|---|---|---|---|---|---|
| LABR | 1 | 557 | 12.9% | 8.4% | 8.6% | 13.1% |
| LABR | 2 | 572 | 26.4% | 15.9% | 17.8% | 26.0% |
| LABR | 4 | 580 | 22.4% | 22.2% | 14.5% | 16.2% |
| LABR | 5 | 574 | 11.8% | 8.0% | 7.0% | 5.2% |
| HARD, descriptor removed | 1 | 1,597 | 9.5% | 6.4% | 6.7% | 7.0% |
| HARD, descriptor removed | 2 | 3,832 | 15.7% | 11.0% | 12.0% | 12.7% |
| HARD, descriptor removed | 4 | 2,555 | 18.4% | 15.3% | 14.4% | 14.9% |
| HARD, descriptor removed | 5 | 2,723 | 5.9% | 3.1% | 3.2% | 3.0% |

- **Border ratings are the hard ones for every system.** MARBERT errs on 17.8% of 2-star and 14.5% of 4-star LABR reviews against 8.6% and 7.0% for 1 and 5 stars, and on HARD without the descriptor 12.0% and 14.4% against 6.7% and 3.2%. This is expected: the 3-star class was removed, so ratings 2 and 4 sit next to the decision boundary.
- **Whether positive or negative labels are harder depends on the cut.** By class, MARBERT errs on 13.3% of gold-negative LABR reviews (150 of 1,129) against 10.7% of gold-positive (124 of 1,154), and on 10.4% (567 of 5,429) against 8.6% (456 of 5,278) on HARD. Negative-labelled rows have the higher error rate on both. 4-star rows err more than 2-star rows on HARD for all four systems (MARBERT 14.4% against 12.0%), but not on LABR for TF-IDF, MARBERT and XLM-R.
- **The hypothesis from the highest-confidence rows is not a general pattern.** It describes those consensus errors (24 of 27 gold-positive), not errors overall.
- **MARBERT's lead over AraBERT on LABR sits in the 4-star rows:** 14.5% against 22.2%. On 2-star rows AraBERT is slightly better (15.9% against 17.8%), and on 1 and 5 stars the two are within 1.0 point. By class, AraBERT misses about 15% of gold-positive LABR reviews (computed from the rounded star-level rates) against 10.7% for MARBERT. One seed each. The cause was not tested. AraBERT's unknown pieces on intensifiers (Chapter 8) are a candidate, though not an obviously class-specific one.

Predictions made before running, scored:
1. **Failed.** Border ratings err at least twice as often as extremes in 9 of 16 checks (LABR 4 against 5 for AraBERT, MARBERT and XLM-R, not TF-IDF at 1.9 times; LABR 2 against 1 for TF-IDF and MARBERT only; HARD 4 against 5 for all four; HARD 2 against 1 for none).
2. **Half held.** 4-star errors exceed 2-star errors on HARD for all four systems, not on LABR for three of four.
3. **Held.** See below.

### Calibration of MARBERT's confidence

Confidence is the larger of the two softmax probabilities. Cells show accuracy, with mean confidence in brackets, and the number of rows. ASTD has no rows at 0.99 or more.

| Dataset | [0.5, 0.7) | [0.7, 0.9) | [0.9, 0.99) | [0.99, 1] | Expected calibration error |
|---|---|---|---|---|---|
| AJGT (180) | 0.000 (0.616), 1 | 0.333 (0.837), 3 | 0.667 (0.968), 9 | 0.946 (0.997), 167 | 0.075 |
| ASTD (242) | 0.636 (0.599), 11 | 0.828 (0.830), 29 | 0.941 (0.972), 202 | none | 0.028 |
| LABR (2,283) | 0.459 (0.612), 37 | 0.627 (0.815), 75 | 0.712 (0.968), 222 | 0.917 (0.997), 1,949 | 0.102 |
| HARD without descriptor (10,707) | 0.517 (0.600), 263 | 0.616 (0.819), 515 | 0.882 (0.978), 4,476 | 0.969 (0.996), 5,453 | 0.066 |

Rows at confidence 0.9 or more: AJGT 176, accuracy 0.932, mean confidence 0.996 (6.4 points below); ASTD 202, 0.941 against 0.972 (3.1); LABR 2,171, 0.896 against 0.994 (9.8); HARD 9,929, 0.930 against 0.988 (5.8).

- **The model is overconfident.** On LABR, rows between 0.7 and 0.99 are right 63% to 71% of the time while the model reports 82% to 97%. 85% of LABR rows and 51% of HARD rows get 0.99 or more, so most answers claim near-certainty.
- **Confidence still carries some information:** errors have lower mean confidence than correct answers (0.943 against 0.987 on LABR).
- **The expected calibration error depends on the binning** (4 bins here, weights by rows). AJGT's low bins hold 1, 3 and 9 rows, so its figure is noisy.
- **These are the single-dataset MARBERT models.** The deployed combined model is checked in Chapter 15.
- **The API returns this number as `confidence` (Chapter 16).** It orders rows, but it is not a probability of being correct, and the README must not say so. Temperature scaling on validation outputs is the standard fix. Not done.


## What went wrong

| Item | Detail |
|---|---|
| Crash on the first listing | `err.sample(25)` failed because AJGT has only 15 errors. Fixed with `min(25, len(err))` |
| An unsupported claim | I said the error rates were similar across dialects. It holds on ASTD only (see above) |
| README wording | It called the annotator "the model author's assistant", which hid that it was an AI. Corrected |
| Categories invented while reading | They were not fixed beforehand, so they may fit this sample better than a fresh one |
| LABR is a sample | 25 of 274 errors |
| A hypothesis from a subset | I suggested that positive labels hide criticism more than negative labels hide praise, from the highest-confidence rows all four systems miss. By class, negative-labelled rows have the higher error rate on LABR and HARD. Retracted above |

## Not done

- Human tagging. `error_sheet.py` writes a blind sheet of the same 61 rows. Compare your category totals with the table. Per-row agreement is not possible, because my tags exist only as counts.
- Errors of the other systems, HARD errors, and errors by class for the LLMs.
- A larger LABR sample.
- Calibration (the confidence figures in `error_stats.py` are a first look, not a calibration study).

## Check yourself

1. Why are the 61 errors not a reliable estimate of dialect-driven failures?
2. What share of the read errors look like label problems, and what does that imply about the ceiling for scores?
3. Why does plain cross-entropy matter for ASTD's missed positives, and how would you test it?
4. Which statement about errors by dialect did I have to correct, and how?
5. How would you design a human re-tagging that tells you the annotator's reliability?
6. Why would an error that all four systems make with high confidence be worth reading?
