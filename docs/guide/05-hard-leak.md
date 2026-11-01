# Chapter 5: The HARD leak

## What

Most HARD reviews begin with a short phrase that almost determines the label. A model can score highly by reading that phrase, which says little about understanding Arabic sentiment. This chapter covers how it was found, how large it is, what was done about it, and what is still uncertain.

## Why it matters

- **Headline integrity.** HARD is the largest dataset in the benchmark. A leaked score of 0.94 would make every model look strong.
- **Deployment.** A real user does not type a Booking.com score label at the start of a review. The leak-free variant is the one that resembles what the deployed API receives.
- **Interview value.** Finding and measuring a data problem is a stronger story than reporting a high score.

## How it was found

1. **First-word table.** After the encoding problem was fixed (Chapter 4), a count of each review's first word against the label showed extreme values. استثنائي was positive in 7,313 reviews and negative in none. ضعيف was negative in 11,957 and positive in 1. مخيب was negative in 9,658 and positive in none.
2. **Hypothesis.** These look like score labels attached to the review text. This is my inference. I did not check it against the HARD paper.
3. **Segment against rating.** Reviews are written as segments separated by periods, so I took the text before the first period and crossed it with the 1, 2, 4, 5 star rating:

| First segment | 1 star | 2 stars | 4 stars | 5 stars | Total | Share at its main rating |
|---|---|---|---|---|---|---|
| مخيب للأمل | 4 | 9,361 | 0 | 0 | 9,365 | 99.96% (2 stars) |
| ضعيف جداً | 4,883 | 7 | 0 | 0 | 4,890 | 99.9% (1 star) |
| ضعيف | 47 | 6,347 | 1 | 0 | 6,395 | 99.2% (2 stars) |
| استثنائي | 0 | 0 | 3 | 6,724 | 6,727 | 99.96% (5 stars) |
| جيد | 17 | 128 | 7,849 | 52 | 8,046 | 97.6% (4 stars) |
| جيد جداً | 1 | 6 | 2,600 | 19 | 2,626 | 99.0% (4 stars) |
| ممتاز | 29 | 18 | 306 | 3,483 | 3,836 | 90.8% (5 stars) |
| رائع | 11 | 17 | 2,667 | 457 | 3,152 | 84.6% (4 stars) |
| مقبول | 9 | 317 | 43 | 1 | 370 | 85.7% (2 stars) |

The phrases carry the star level, not just the polarity. The pattern is consistent with a site's rating categories. I did not verify this.

4. **Lookup test (my arithmetic from the table).** Across these nine phrases, 45,407 reviews (about 44% of 104,036 deduped reviews) carry one. Labeling each by the majority polarity of its phrase gives 45,135 correct, which is 99.4%. A nine-word lookup table labels 44% of the dataset almost perfectly.
5. **A test that looked negative and was incomplete.** I suspected that a leading quote marks user-typed text and tested only the curly quote. It opened 53.5% of reviews at similar rates across ratings, so I dropped the idea. That test counted reviews opening with a straight quote as unquoted, which hid the pattern. With both quote types the signal is almost perfect (next section).
6. **Size of the effect, on a first diagnostic.** TF-IDF with logistic regression, random 80/20 split, deduplicated text: macro-F1 0.9369 on raw text and 0.8395 with everything before the first period removed (rows with a remainder of at least 3 characters). The second number is pessimistic, because that crude rule also deletes genuine user text.

## The structure (checked on the whole dataset)

Measured on all 103,034 rows of `hard.parquet`, counting both curly and straight quotes:

| | Starts with a quote | Starts without a quote |
|---|---|---|
| First segment is one of the nine phrases | 6,185 | 38,222 |
| First segment is anything else | 58,627 | 0 |

- Every row is either quoted (64,812 rows, 62.9%) or opens with one of the nine phrases and no quote (38,222 rows, 37.1%). No unquoted row opens with anything else.
- So the first segment is either a quoted user title or an unquoted phrase from the list. The unquoted phrases are the leak. That they are Booking.com score categories shown in place of a missing title is still my inference, not checked against the HARD paper.
- A first segment inside quotes is a genuine user title even when it equals a descriptor word. Row 0 of the raw file is a 2-star review titled “ممتاز”.
- 6,185 matched rows are quoted. That is 6.0% of all rows and 13.9% of the 44,407 matched rows. Quoted share by phrase: ممتاز 34.8%, رائع 26.1%, جيد 16.1%, جيد جداً 11.4%, ضعيف 10.4%, مخيب للأمل 8.1%, استثنائي 7.2%, ضعيف جداً 6.5%, مقبول 100% (all 363 rows).
- مقبول is therefore never an unquoted label. Its apparent star pattern in the first table (86% at 2 stars, counted on 370 rows before a later dedupe) came from user titles.
- The rule used in all experiments removes quote marks before comparing, so it strips these 6,185 user titles as well.

Why the earlier quote test missed this: it counted only the curly quote, so rows opening with a straight quote looked unquoted.

## The fix: two variants

| Variant | Text | Use |
|---|---|---|
| `hard` | as released | comparable with published HARD work, inflated |
| `hard_notitle` | first segment removed when it exactly equals one of nine phrases | the better measure of sentiment understanding, and closer to deployment input |

Both use identical rows and identical splits, so the score difference measures the leak and nothing else (Chapter 4 explains how).

## Alternatives

| Option | Verdict |
|---|---|
| Drop HARD | Loses the largest dataset and the finding itself |
| Use only the cleaned variant | Hides how the leak behaves and cannot be compared with the literature |
| Remove the first segment of every review | Also deletes real user titles and removes sentiment that is legitimate. This was the crude diagnostic, and its 0.8395 is too pessimistic |
| Remove the descriptor only when unquoted (strict rule) | Matches the structure above and needs no phrase list. Tested for TF-IDF only (see the strict-rule check below): +0.56 macro-F1 against the rule used. Not applied to the data or to the encoder and LLM runs |
| Mask the phrase with a placeholder token | Keeps the sentence shape. Not tried |
| Detect the leak with a model (a classifier on the first segment) | More complex than an exact-match list, and I did not try it |
| Train on `hard`, test on `hard_notitle` | A useful extra experiment to show how much models rely on the shortcut. Not run |

## How: the diagnostics

The first-word and segment tables are pandas crosstabs over the raw file. The TF-IDF measurement:

```python
df = df.drop_duplicates("review")
df["seg"] = df.review.str.split(".", n=1).str[0].str.strip(" “”\"")
df["rest"] = df.review.str.split(".", n=1).str[1].fillna("").str.strip()
d = df[df.rest.str.len() >= 3]
tr, te = train_test_split(d, test_size=0.2, stratify=d.y, random_state=42)
for name, col in [("raw review", "review"), ("descriptor removed", "rest")]:
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)
    m = LogisticRegression(max_iter=1000).fit(v.fit_transform(tr[col]), tr.y)
    p = m.predict(v.transform(te[col]))
    print(name, f1_score(te.y, p, average="macro"))
```

- `str.split(".", n=1)` splits only at the first period, giving the segment and the rest.
- The same train and test rows are used for both columns, so only the text differs.
- `n=1` matters: without it the "rest" would be cut at the next period too.

The final rule is `strip_descriptor` in Chapter 4.

## What went wrong, and what is still weak

| Item | Detail |
|---|---|
| Crude diagnostic | Removing everything before the first period gave 0.8395, which exaggerates the leak because it deletes real text (for example "فندق رائع" or "لا انصح به", which carry genuine sentiment) |
| First-word table was not a coverage figure | I reported that ten first words covered about 57% of rows, but four of those words (الفندق, فندق, تجربة, لا) are ordinary openings and not descriptors. The descriptor coverage figure is about 44% |
| "Conservative lower bound" overreached | I called `hard_notitle` a conservative lower bound. The rule over-strips 6.0% of rows (quoted user titles), so it is not a bound in either direction. The other rows start with a quoted user title, which is correct to keep |
| Hypothesis unverified | That the phrases are Booking.com score labels and that they replace a missing user title are inferences. Neither was checked against the paper |
| Random split versus final split | The diagnostic used a random split. The final numbers use hotel-level splits, so 0.9369 and 0.8395 are not the same quantities as the later 0.9411 and 0.8710 |

## Numbers

Leak gap on the 10,707-review full test set (single seed for encoders, trained on a 20,000-review subsample):

| System | `hard` | `hard_notitle` | Drop |
|---|---|---|---|
| TF-IDF + LR | 0.9411 | 0.8710 | 7.0 |
| AraBERT | 0.9568 | 0.9067 | 5.0 |
| MARBERT | 0.9561 | 0.9045 | 5.2 |
| XLM-R | 0.9530 | 0.9009 | 5.2 |

Reading: TF-IDF loses more than the encoders (7.0 against 5.0 to 5.2), a difference well outside the plus or minus 0.6 interval of each score, though the encoders are single-seed. A plausible reason is that encoders extract more sentiment from the body text, which I did not test.

On the 300-row comparison sample (confidence intervals about plus or minus 4 points, so do not rank these):

| System | `hard` | `hard_notitle` | Drop |
|---|---|---|---|
| TF-IDF | 0.950 | 0.850 | 10.0 |
| AraBERT | 0.953 | 0.927 | 2.7 |
| MARBERT | 0.960 | 0.907 | 5.3 |
| XLM-R | 0.953 | 0.893 | 6.0 |
| Qwen | 0.927 | 0.862 | 6.4 |
| gpt-oss | 0.866 | 0.785 | 8.2 |
| ALLaM | 0.927 | 0.876 | 5.1 |

Every system type loses points when the descriptor is removed, including the few-shot LLMs.

## Strict-rule check

Script: `src/arabic_sentiment/hard_strict_check.py`. TF-IDF + logistic regression with the configuration selected for `hard_notitle` (normalization on, character n-grams, C=10), trained and tested on the same hotel-level split. The strict rule leaves a phrase alone when the review starts with a quote. It leaves 4,914 training rows and 608 of the 10,707 test rows unstripped.

| Variant | Macro-F1 | Accuracy on the 608 quoted-match rows | Accuracy on all other rows |
|---|---|---|---|
| raw text | 0.9410 | 0.961 | 0.940 |
| current rule (used in all experiments) | 0.8710 | 0.842 | 0.873 |
| strict rule | 0.8766 | 0.952 | 0.872 |

- The `current` line reproduces the saved 0.8710. The `raw text` line is 0.9410 against the saved 0.9411 because the saved baseline selected C=1 for the raw variant.
- All of the +0.56 comes from the 608 rows. Where the current rule deletes the user's own title, accuracy falls from 0.961 to 0.842, and the strict rule restores it to 0.952. Other rows are unchanged (0.873 and 0.872).
- Leak gap for TF-IDF: 7.0 points under the current rule, 6.4 under the strict rule.

Decision (rule fixed before running: keep the data if the difference is under about 1 point): keep the data and every result, and document the flaw. Not done: the encoders and LLMs were not rerun with the strict rule, so their sensitivity is unknown, and no paired significance test was run on the overall difference. `data.py` was not changed, because the processed data and all results correspond to the current rule. If the project were redone, the strict rule is the one to use.

## Check yourself

1. How can a model score 0.94 on HARD while understanding little Arabic?
2. What does the table of segment against star rating show beyond polarity?
3. Why was the curly-quote test not enough to decide anything about the descriptor?
4. Why is the crude diagnostic's 0.8395 too pessimistic?
5. Why do `hard` and `hard_notitle` use identical rows and splits?
6. Which variant is closer to what the deployed API receives, and why?
7. Name two ways `hard_notitle` can still be wrong.
