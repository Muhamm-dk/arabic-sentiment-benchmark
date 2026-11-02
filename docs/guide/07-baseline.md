# Chapter 7: The baseline

## What

TF-IDF features with logistic regression, tuned on validation and scored once on test, for each of the five dataset variants. Code: `src/arabic_sentiment/baseline.py` (selection and scoring) and `baseline_preds.py` (saves test predictions).

## Why a baseline at all

Without a simple, well-tuned reference, a transformer's score has no meaning. "0.88 on LABR" tells you nothing. "0.88 against 0.8155 for TF-IDF" tells you what fine-tuning bought. A weak baseline inflates every later gain, so the baseline gets a fair search too (12 configurations per dataset).

Why TF-IDF with logistic regression specifically:
- Cheap: the whole search took about 7 minutes (from timestamps of two runs).
- Strong on text classification with enough data.
- Deterministic and interpretable: the learned weights show which words and character sequences drive predictions.
- No GPU.

## How TF-IDF works, in the form sklearn computes it

1. Each text becomes a vector with one entry per term (a word, or a character sequence).
2. The raw count `tf` is damped by `sublinear_tf=True`: it becomes `1 + ln(tf)`, so a word repeated ten times does not count ten times.
3. Each term is multiplied by an inverse document frequency: `ln((1 + n) / (1 + df)) + 1` (sklearn's default smoothed form), so terms that appear in many documents count less.
4. Each row is scaled to unit length (L2 norm).

A logistic regression then learns one weight per term. The prediction is the sign of the weighted sum plus a bias, passed through a sigmoid for a probability.

## The search space

```python
for norm, kind in product([False, True], ["word", "char"]):
    ...
    for C in (1, 10, 100):
```

2 normalization settings x 2 feature types x 3 values of C = 12 fits per dataset, 60 in total.

### Feature type 1: word 1-2 grams

```python
TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
```

- `ngram_range=(1, 2)`: single words and adjacent word pairs (so "ليس جيد" is one feature, which helps with negation).
- `min_df=2`: a term must appear in at least two training texts, which removes one-off typos.
- `max_features=300_000`: keep the 300,000 most frequent terms. Whether this cap was reached was not checked.
- Defaults I did not change: lowercasing, and a word pattern that ignores one-character tokens. A standalone one-letter Arabic word is therefore dropped. I believe Python's `\w` does not match combining diacritics, so unremoved diacritics may split words; I did not test it.

### Feature type 2: character n-grams

```python
TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000)
```

- `char_wb` builds character sequences of length 2 to 5 only inside word boundaries, padding each word with a space on both sides.
- Why it suits Arabic: attached prefixes and suffixes (و, ب, ال, ها and similar) and dialect spelling variants break word tokens into many rare forms, while character sequences still overlap between variants. This is a rationale, not something tested with a stemmer.
- Cost: the feature space is large and the weights are harder to read.

### Normalization on or off

```python
TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
REPEAT = re.compile(r"(.)\1{2,}")

def normalize(t):
    t = TASHKEEL.sub("", t).replace("\u0640", "")
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي")
    return REPEAT.sub(r"\1\1", t)
```

| Step | What it does | Why |
|---|---|---|
| `TASHKEEL.sub("", t)` | removes Arabic diacritics (ranges U+0610 to U+061A, U+064B to U+065F, U+0670, U+06D6 to U+06ED) | diacritics are optional in writing, so the same word appears with and without them |
| `.replace("\u0640", "")` | removes tatweel, the stretching character | purely decorative |
| `re.sub("[إأآٱ]", "ا", t)` | maps the alef-with-hamza forms and alef wasla to plain alef | writers are inconsistent with hamza |
| `.replace("ى", "ي")` | maps alef maqsura to yeh | commonly interchanged in informal writing, especially Egyptian text |
| `REPEAT.sub(r"\1\1", t)` | any letter repeated three or more times becomes two | shrinks elongation such as "جميييييل". It does not restore the standard spelling |

Deliberately not done: mapping ة to ه (this merges different words), removing punctuation, hashtags, URLs, digits or emoji, stemming or lemmatizing, and anything for Latin-script text.

## Alternatives

None of these were run.

| Choice | Alternative | Note |
|---|---|---|
| Logistic regression | Linear SVM, SGD classifier, Complement or Multinomial Naive Bayes, NB-SVM | Usually close to logistic regression on this kind of task |
| | Gradient boosting | Usually worse on sparse text |
| TF-IDF | Raw counts, binary features, hashing | Likely small differences |
| | fastText | A strong, fast character-aware baseline |
| | Pretrained static word vectors (AraVec and similar) with a classifier | Adds semantic similarity without fine-tuning |
| | Lexicon-based sentiment scoring | Needs an Arabic lexicon, not used |
| Normalization function above | Farasa or CAMeL Tools morphological analysis, stemming, lemmatization | Handles prefixes and suffixes properly, but needs extra tools and I did not try them |
| Grid of 12 | Random search, a finer C grid | Cheap here, so a finer grid is easy |
| `liblinear` | `lbfgs`, `saga` | `lbfgs` is the default for multinomial problems. liblinear is fast for sparse binary problems |

## The selection loop

```python
txt = {s: (d.text.map(normalize) if norm else d.text) for s, d in parts.items()}
v = vectorizer(kind)
X = {"train": v.fit_transform(txt["train"])}
X["val"], X["test"] = v.transform(txt["val"]), v.transform(txt["test"])
for C in (1, 10, 100):
    m = LogisticRegression(C=C, solver="liblinear", max_iter=1000).fit(X["train"], parts["train"].label)
    f1_val = f1_score(parts["val"].label, m.predict(X["val"]), average="macro")
    if best is None or f1_val > best["val_macro_f1"]:
        best = dict(norm=norm, feat=kind, C=C, val_macro_f1=f1_val, model=m, Xtest=X["test"])
```

- The vectorizer is fitted on the training text only (`fit_transform` on train, `transform` on val and test). Fitting on all text would leak the validation and test vocabulary and document frequencies.
- The vectorizer is fitted once per (normalization, feature type) and reused across the three values of C.
- `C` is the inverse of regularization strength: small C means strong regularization.
- `liblinear` solves the binary problem with a coordinate-descent method. It also penalizes the intercept (a documented sklearn note, from memory).
- `max_iter=1000` was enough that no convergence warning was raised in the runs I saw.
- The comparison is strict (`>`), so a tie keeps the first configuration in loop order.
- The test matrix is stored with the best model, so the test set is predicted once, after all 12 configurations were scored on validation.

## Predictions, and why a second script exists

`baseline.py` originally saved only scores. The LLM comparison uses 300-row subsets of LABR and HARD, so TF-IDF had to be scored on exactly those rows. `baseline_preds.py` reads the chosen configuration from `baseline.json`, refits on the training split alone, and saves test predictions as `results/preds/<dataset>-tfidf-s0.parquet`.

It prints the refit score next to the saved one. All five matched to four decimals (LABR 0.8155, HARD 0.9411, HARD no-descriptor 0.8710, ASTD 0.7708, AJGT 0.8722), which shows the fit is deterministic here.

## Numbers

### Selected configurations (12 configurations scored on validation each)

| Dataset | Normalize | Features | C | Val macro-F1 | Test macro-F1 | 95% CI | Test accuracy | n test |
|---|---|---|---|---|---|---|---|---|
| LABR | no | char | 10 | 0.8395 | 0.8155 | [0.7986, 0.8304] | 0.8156 | 2,283 |
| HARD | yes | char | 1 | 0.9411 | 0.9411 | [0.9366, 0.9454] | 0.9411 | 10,707 |
| HARD no-descriptor | yes | char | 10 | 0.8753 | 0.8710 | [0.8648, 0.8772] | 0.8710 | 10,707 |
| ASTD | no | char | 10 | 0.8190 | 0.7708 | [0.7095, 0.8277] | 0.8182 | 242 |
| AJGT | yes | char | 10 | 0.8608 | 0.8722 | [0.8217, 0.9166] | 0.8722 | 180 |

### Best validation macro-F1 by feature type

| Dataset | Best word config | Best char config | Gap |
|---|---|---|---|
| LABR | 0.8364 | 0.8395 | 0.3 points |
| HARD | 0.9387 | 0.9411 | 0.2 |
| HARD no-descriptor | 0.8659 | 0.8753 | 0.9 |
| ASTD | 0.7568 | 0.8190 | 6.2 |
| AJGT | 0.8440 | 0.8608 | 1.7 |

Character n-grams win on every dataset. The gap is small on the large review datasets and large on the tweet datasets, where spelling variation is greatest. This is consistent with the rationale above, though a single comparison cannot prove it.

### Does normalization help?

It changes validation scores by under 1 point in most cases and its sign flips between datasets. Selected: on for HARD, HARD no-descriptor and AJGT, off for LABR and ASTD. On ASTD with character features, C=10: 0.8190 without, 0.8027 with. I treat it as neutral. The guide does not claim it helps.

### The C grid

| Dataset | Selected C | Neighbours (same features) |
|---|---|---|
| LABR | 10 | 0.8342 (C=1), 0.8395, 0.8233 (C=100) |
| ASTD | 10 | 0.6710, 0.8190, 0.7946 |
| AJGT | 10 | 0.8547, 0.8608, 0.8386 |
| HARD no-descriptor | 10 | 0.8751, 0.8753, 0.8544 |
| **HARD** | **1** | **0.9411, 0.9391, 0.9270** |

The winner is interior except for raw HARD, where C=1 is the lowest value tried. A smaller C was never tested there, and could score slightly higher. The effect on the conclusions should be small (the C=1 and C=10 scores differ by 0.2 points, and the leak gap is 7 points), but it is an open edge.

### Validation against test

LABR val 0.8395 against test 0.8155 (2.4 points lower), ASTD 0.8190 against 0.7708 (4.8 lower), AJGT 0.8608 against 0.8722 (1.1 higher), HARD equal, HARD no-descriptor 0.4 lower. Two causes are plausible: picking the best of 12 on a small validation set is optimistic, and the LABR validation set is a random slice of the training data while the test set is the official one. I did not test either.

## What went wrong

| Item | Detail |
|---|---|
| C grid too narrow at first | The first run used C in {1, 10}. ASTD's best sat at the edge (0.671 for C=1, 0.819 for C=10), so a baseline that weak would have inflated every later gain. I extended it to {1, 10, 100}. No winner changed. I wrongly said the optimum was interior everywhere (see HARD above) |
| No predictions saved | Fixed later by `baseline_preds.py` |
| Wrong comparison claim | I said word features lose 1 to 3 points to characters. Best against best, it is 0.2 to 6.2 points depending on the dataset |
| Two different TF-IDF numbers for HARD | Chapter 5's quick diagnostic used word 1-2 grams, a random split and default C, so its 0.9369 and 0.8395 are not the same quantities as the final 0.9411 and 0.8710 |
| `max_features` | Not checked whether the cap truncated the vocabulary on HARD |

## Try it yourself (not run)

To see the leak inside the model, fit a word-level model on HARD training text and print the largest weights:

```python
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
d = pd.read_parquet("data/processed/hard.parquet"); tr = d[d.split == "train"]
v = TfidfVectorizer(ngram_range=(1, 1), min_df=5, sublinear_tf=True)
m = LogisticRegression(C=10, solver="liblinear").fit(v.fit_transform(tr.text), tr.label)
terms = np.array(v.get_feature_names_out()); w = m.coef_[0]
print("negative:", terms[np.argsort(w)[:15]])
print("positive:", terms[np.argsort(w)[-15:]])
```

I would expect the descriptor words near the top of both lists, but I have not run it. If they are not, the leak is spread over other features.

## Check yourself

1. Why must the vectorizer be fitted on the training text only?
2. What does `sublinear_tf=True` do, and why does it help on long reviews?
3. Why would character n-grams suit Arabic better than word tokens, and what does the 6.2-point ASTD gap suggest?
4. Why is the claim "normalization helps" not supported by these results?
5. HARD's best C is the lowest value in the grid. What does that mean, and what would you run?
6. ASTD's test macro-F1 is 4.8 points below validation. Give two explanations.
7. Why did the baseline get a second script, and what did it check?
8. Why does the strict `>` in the selection loop matter?
