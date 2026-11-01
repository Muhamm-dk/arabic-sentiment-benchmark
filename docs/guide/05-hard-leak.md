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
5. **A rejected hypothesis.** I suspected that a leading curly quote marked user-typed text. It did not: 53.5% of reviews start with a quote, at similar rates across ratings (1 star: 59.2%, 2 stars: 52.6%, 4 stars: 47.6%, 5 stars: 57.5%). This test only ruled out quote use as an overall signal. It did not test whether descriptors are unquoted (see the open check below).
6. **Size of the effect, on a first diagnostic.** TF-IDF with logistic regression, random 80/20 split, deduplicated text: macro-F1 0.9369 on raw text and 0.8395 with everything before the first period removed (rows with a remainder of at least 3 characters). The second number is pessimistic, because that crude rule also deletes genuine user text.

## The structure hypothesis

Printed samples looked like `"title". liked part. disliked part.`, for example `ضعيف جداً. . موقع المكان غير مناسب عند الخروج مزدحم` (empty liked part) or `“بو محمد”. الموقع. ازعاج...`. In all 21 printed rows, the quoted first segment was a user-typed title and the unquoted one was a score-like word. If a user title exists, it is shown in quotes, and otherwise the score word takes its place.

This fits the numbers: about 44% of reviews match a descriptor, and 46.5% of reviews start without a quote (100% minus the 53.5% that start with one). It is a hypothesis from a sample, not a result.

Consequences if true:
- A leading segment inside quotes is a genuine user title, even when it equals a descriptor word (row 0 of the raw file is a 2-star review titled “ممتاز”).
- Our rule removes quote marks before comparing, so it also strips user titles that equal a descriptor word.
- Rows with a user title carry no descriptor, so they keep whatever sentiment the title holds.

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
| Remove the descriptor only when unquoted | Probably the most accurate rule if the structure hypothesis holds. Not applied yet (open check) |
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
| "Conservative lower bound" overreached | I called `hard_notitle` a conservative lower bound. It can remove genuine user titles, and the other 56% of rows keep their leading segment, so it is neither a clean measure nor a strict bound |
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

## Open check

Test the structure hypothesis: among rows whose first segment matches a descriptor, what share start with a quote? Run this and write the result below.

```python
from arabic_sentiment.data import DESCRIPTORS, lead
import pandas as pd
d = pd.read_parquet("data/processed/hard.parquet")
hit = d.text.map(lead).isin(DESCRIPTORS)
quoted = d.text.str.lstrip().str.startswith(("\u201c", '"'))
print(pd.crosstab(hit, quoted, rownames=["descriptor match"], colnames=["starts with quote"]))
print(quoted[hit].groupby(d.text[hit].map(lead)).agg(["size", "mean"]).round(3))
```

Result: (fill in).

## Check yourself

1. How can a model score 0.94 on HARD while understanding little Arabic?
2. What does the table of segment against star rating show beyond polarity?
3. Why was the curly-quote test not enough to decide anything about the descriptor?
4. Why is the crude diagnostic's 0.8395 too pessimistic?
5. Why do `hard` and `hard_notitle` use identical rows and splits?
6. Which variant is closer to what the deployed API receives, and why?
7. Name two ways `hard_notitle` can still be wrong.
