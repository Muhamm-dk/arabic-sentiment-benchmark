# Chapter 13: Dialect analysis

## What

The brief asked for an error analysis focused on dialect failures. The question here: does sentiment accuracy differ between Modern Standard Arabic text and dialect text, and between particular dialects? Two scripts do the work: `dialect.py` (tags test rows with a predicted dialect) and `dialect_analysis.py` (scores each system within each dialect group). `dialect_checks.py` adds two checks.

Short answer, with the caveats below: the encoders show **no gap beyond noise** on LABR and HARD, and one candidate effect (XLM-R on Egyptian tweets) that this data cannot confirm. The classifier behind the groups is third-party and was never validated on our data, which limits what a null result means.

## The central problem: no dialect labels

None of the four datasets has per-row dialect labels. The `variety` column is one value per dataset (`mixed`, `egyptian`, `jordanian`), and a dataset is not a dialect: ASTD's tweets are partly MSA, and LABR and HARD mix both. So the groups have to be predicted.

| Option | Verdict |
|---|---|
| Dataset as the dialect proxy (AJGT = Jordanian, ASTD = Egyptian) | Free, but coarse and confounded with domain (tweets against reviews). Used only as context |
| **Predicted label per row by a pretrained classifier** | Chosen. Per-row, but third-party and unvalidated here |
| Hand-built word lists (for example Egyptian "ازاي", Levantine "هلق") | Transparent, but covers only a minority of rows. Not built |
| Human annotation of a sample | The proper fix. Not done. About 100 rows per dataset would let you measure the classifier's accuracy on our data |
| Train our own classifier on a dialect corpus | More work. I did not check which corpora allow this |
| A dialect tool from the Arabic NLP literature (for example CAMeL Tools' dialect identification) | Possible. Not tried, and I did not check how it would load here |

## Choosing the classifier

A hub search found only small community uploads (all under 400 downloads), none from a research group, and I could not judge their quality from the list. A shortlist was inspected by loading each and tagging 15 rows from three datasets.

| Candidate | Labels | Outcome |
|---|---|---|
| `IbrahimAmin/marbertv2-arabic-written-dialect-classifier` | MSA, EGY, LEV, GLF, MAGHREB | **Used.** Fine-tuned from MARBERTv2 (per its card). Has an MSA class, which LABR and HARD need |
| `oahmedd/MARBERTv2-Finetuned-on-5-Dialects-QADI` | Gulf, Egyptian, Levantine, Mesopotamian, Maghrebi | **Second opinion.** Trained on QADI tweets (per its card). No MSA class, so it labelled formal LABR reviews Maghrebi or Levantine |
| `jabalov/MARBERT-finetuned-arabic-dialects-identification` | not seen | Failed to load (a TensorFlow checkpoint). Its card says "unknown dataset" and one epoch |
| `Ammar-alhaj-ali/arabic-MARBERT-dialect-identification-city` | city level | Skipped as too fine-grained for datasets this size |

What I did not check: the training data and reported accuracy of the chosen classifier (my read of its card was cut off), and how it performs on tweets and reviews like ours. The 15-row check gave two warnings: very short Jordanian tweets (for example "قلت ادب") got low-confidence labels, and several formal LABR reviews were labelled MAGHREB at 0.49 to 0.88 confidence, though they read as MSA.

## How: tagging

```python
def tag(model, texts):
    clf = pipeline("text-classification", model=model, device=0, truncation=True, max_length=128,
                   torch_dtype=torch.bfloat16)
    out = clf(texts, batch_size=64)
    del clf
    torch.cuda.empty_cache()
    return [o["label"] for o in out], [round(o["score"], 4) for o in out]
```

- Only test rows are tagged (13,412 texts across the four datasets), by both classifiers, in the order of the processed files.
- Text is truncated at 128 tokens, so for long reviews only the opening decides the label.
- `torch_dtype` triggers a deprecation warning in the current library (`dtype` replaces it). It still ran.
- The score is the classifier's softmax probability for its top label, rounded to four places. It is not calibrated. A 0.9 does not mean 90% accuracy.
- `hard_notitle` is not tagged separately: it shares rows with `hard`, so the `hard` tags (made on the raw text, descriptor included) are used for both. That means a classifier may have seen the descriptor word when labelling.
- Output: `results/dialect_tags.parquet`, not in git (it contains dataset texts). The first version of the script was corrupted by a pasted heredoc (a stray line after `main()`) and was repaired with `sed -i`.

### What the tags look like (test rows)

| Dataset (n) | EGY | GLF | LEV | MAGHREB | MSA | Confidence at least 0.7 |
|---|---|---|---|---|---|---|
| AJGT (180) | 20 | 47 | 60 | 17 | 36 | 67% |
| ASTD (242) | 107 | 6 | 11 | 2 | 116 | 91% |
| HARD (10,707) | 80 | 3,045 | 840 | 419 | 6,323 | 80% |
| LABR (2,283) | 488 | 231 | 153 | 257 | 1,154 | 77% |

Sanity reading:
- **ASTD looks right:** Egyptian and MSA together are 92% of rows, matching a mostly Egyptian tweet source. This is the best validation the classifier gets.
- **AJGT does not:** it is Jordanian (Levantine), yet only 60 of 180 tweets (33%) are tagged LEV, with 47 GLF and 17 MAGHREB. On short tweets the classifier is unreliable. AJGT is therefore analysed only as a whole.
- **HARD's 28% Gulf share is not a dialect finding.** Mean length is similar across groups (GLF 25.1 words, MSA 23.9), so short text is not the driver. A read of 12 confident GLF rows found few real Gulf dialect words and several Gulf place names and currencies (شارع زايد, مول الامارات, ريال). So the label there tracks hotel location, which the hotel-level split also tracks. That is my reading of 12 rows.
- **MAGHREB rows are short** (mean 11.8 words) with the lowest mean confidence (0.71).

Mean confidence by predicted label on HARD: EGY 0.78, GLF 0.82, LEV 0.77, MAGHREB 0.71, MSA 0.90.

## How: the analysis

```python
tags_all = pd.read_parquet(ROOT / "results/dialect_tags.parquet")
t = tag_frame(tags_all, d)
t = t[t.conf >= MIN_CONF]                       # 0.7
m = p.merge(t[["text", "dialect"]], on="text", how="inner")
if d == "ajgt":    groups = {"all confident rows": m}
elif d == "astd":  groups = {g: m[m.dialect == g] for g in sorted(m.dialect.unique())}
else:              groups = {"MSA": m[m.dialect == "MSA"], "non-MSA": m[m.dialect != "MSA"]}
groups["ALL (any confidence)"] = p
```

- **Systems:** the three encoders (majority vote over seeds on ASTD and AJGT, the single run elsewhere) and ALLaM (on its own rows: all of AJGT and ASTD, the 300-row samples of LABR and HARD).
- **Confidence cut at 0.7:** noisy labels stay out. Cost: it selects rows. On AJGT the confident rows score lower than all rows (MARBERT 0.891 against 0.917), so the selection is not neutral.
- **Groups:** MSA against non-MSA on the review datasets. ASTD by predicted dialect. AJGT undivided.
- **Minimum group size 50.** Smaller groups print `small (n=...)`.
- **Intervals:** 500 bootstrap resamples per group (the rest of the guide uses 1,000 or 2,000).

### The first result table

| Dataset | Group | AraBERT | MARBERT | XLM-R | ALLaM (own rows) |
|---|---|---|---|---|---|
| LABR | MSA (n 1,006; ALLaM 132) | 0.866 | 0.871 | 0.856 | 0.880 |
| LABR | non-MSA (n 745; ALLaM 93) | 0.868 | 0.890 | 0.841 | 0.858 |
| HARD | MSA (n 5,522; ALLaM 161) | 0.955 | 0.954 | 0.952 | 0.907 |
| HARD | non-MSA (n 3,044; ALLaM 83) | 0.956 | 0.955 | 0.952 | 0.938 |
| ASTD | EGY (n 105) | 0.895 | 0.880 | 0.809 | 0.724 |
| ASTD | MSA (n 103) | 0.893 | 0.905 | 0.903 | 0.772 |
| AJGT | all confident rows (n 120) | 0.880 | 0.891 | 0.864 | 0.851 |

Other ASTD groups (GLF 5, LEV 7, MAGHREB 1) are too small. Intervals are in `results/dialect_analysis.csv`: about ±2 to 3 points on LABR, ±1 on HARD, ±7 to 9 on ASTD.

The earlier `hard_notitle` rows of this table were invalid (see "What went wrong"), so they are omitted here.

## Reading it

1. **LABR and raw HARD:** MSA and non-MSA are level for AraBERT and MARBERT. The largest visible gap is MARBERT on LABR (+1.9 in favour of non-MSA), with overlapping intervals.
2. **ASTD:** XLM-R is 9.4 points lower on Egyptian tweets than on MSA ones (0.809 against 0.903). The others show small differences. It is the only candidate effect and the groups hold about 100 rows each, so I call it suggestive. It also matches XLM-R's weaker results on tweets generally (Chapter 10).
3. **ALLaM** is no worse on non-MSA than MSA on LABR and HARD, but its group sizes (83 to 161 rows) make that uninformative.
4. **A manual read of 61 MARBERT errors** (Chapter 14) found dialect vocabulary plausible in about 13% of them. Most errors were missing or mixed sentiment, stance-dependent labels and questionable gold labels.

## What a null result can and cannot mean

- **Label noise pushes toward 0.** If the classifier mislabels many rows, any real difference between dialects is blurred. AJGT shows this is not hypothetical. A null result may therefore partly come from the tool.
- **Confounding.** Dialect correlates with dataset, domain, text length and (on HARD) hotel region. A gap, or its absence, cannot be attributed to dialect alone. The wording must be "errors do or do not concentrate in these groups", never "dialect causes errors".
- **Confidence selection** removes 20 to 33% of rows, and those rows are not random.
- **Power.** The intervals rule out gaps beyond about 1.2 points on HARD and 2.5 on LABR, but only about 7 on ASTD, so no small effect can be excluded there.
- **Multiple comparisons.** Many group-by-system cells were examined with no correction.
- **Shared ancestry (untested speculation).** The dialect classifier and the sentiment encoders are both MARBERT-family, so rows the tagger finds hard may be rows the sentiment model also finds hard.
- **Coverage.** Almost no Maghrebi or Gulf text in the real sense. Gulf is the most relevant for a UAE audience and this benchmark cannot speak to it.

## What went wrong

| Item | Detail |
|---|---|
| **Wrong `hard_notitle` rows** | The tags were made on the raw `hard` text, but the script joined them to `hard_notitle` predictions by exact text. About 43% of the rows have stripped text and no match, so they dropped silently. What remained was the rows with a quoted user title (n 3,308 and 1,737, a selected subset). I first explained the smaller n by "confidence changing after stripping", which was wrong. My check that the "ALL" row matched did not test the join. Fix: `tag_frame` aligns the `hard` tags to the `hard_notitle` test rows by position and asserts the alignment |
| **Second classifier never used** | QADI tagged every row, but the analysis script did not read its labels. The planned agreement check was never run. `dialect_checks.py` now does it |
| **Corrupted script** | The first `dialect.py` had a pasted heredoc spliced into it. It ran because the stray line came after `main()`. Repaired with `sed -i '44,$d'` |
| **Null result read from overlapping intervals** | Fixed with direct group-difference intervals in `dialect_checks.py` |
| **Overconfident wording** | The README's "no dialect gap beyond noise" needs the caveats above. It is updated below |

## Numbers still to fill in

After running `dialect_checks.py` and the patched `dialect_analysis.py`:

| Item | Result |
|---|---|
| Agreement of the two classifiers on non-MSA rows (ASTD / HARD / AJGT / LABR) | (fill in) |
| `hard_notitle` group sizes after the fix (expected 5,522 and 3,044) | (fill in) |
| `hard_notitle` MSA against non-MSA, per encoder | (fill in) |
| Group-difference intervals, LABR, HARD, ASTD | (fill in) |

## Not done

- Human annotation of a sample to measure the classifier's accuracy on our data.
- Any analysis of Gulf or Maghrebi text.
- A regression with covariates (length, class, dataset) to separate dialect from other factors.
- Calibration of the tagger's confidence.
- A different classifier or a hand-built word-list check.

## Check yourself

1. Why are dialect labels predicted here, and what does that do to the meaning of a null result?
2. Why was the second classifier a poor primary choice, and what was it supposed to add?
3. What does the HARD Gulf label seem to track, and why is that a problem for dialect claims?
4. Why were the `hard_notitle` dialect rows wrong, and why did the check "the ALL row matches" not catch it?
5. Why is a 0.7 confidence cut not neutral, and what evidence do we have?
6. Why does misclassification bias group differences toward 0?
7. What is the one candidate effect, why is it not confirmed, and what would confirm it?
8. What is the right wording of the dialect finding for the README?
