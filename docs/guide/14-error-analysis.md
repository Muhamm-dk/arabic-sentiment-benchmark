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

Predictions made before running:
1. At least 35% of MARBERT's errors are shared with at least two other systems, on every dataset.
2. MARBERT is less confident on its errors, yet more than half of its errors have confidence 0.9 or more on AJGT, ASTD and LABR.
3. Error rate rises with length on LABR and HARD without the descriptor, and is flat on the tweet sets.

Results: (fill in after running).

## What went wrong

| Item | Detail |
|---|---|
| Crash on the first listing | `err.sample(25)` failed because AJGT has only 15 errors. Fixed with `min(25, len(err))` |
| An unsupported claim | I said the error rates were similar across dialects. It holds on ASTD only (see above) |
| README wording | It called the annotator "the model author's assistant", which hid that it was an AI. Corrected |
| Categories invented while reading | They were not fixed beforehand, so they may fit this sample better than a fresh one |
| LABR is a sample | 25 of 274 errors |

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
