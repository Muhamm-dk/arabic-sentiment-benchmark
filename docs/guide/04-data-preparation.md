# Chapter 4: Data preparation

## What

Turn four differently shaped sources into one schema, one label definition and one split discipline, in one script (`src/arabic_sentiment/data.py`) that can be rerun from scratch.

Output: five parquet files in `data/processed/`: `labr`, `hard`, `hard_notitle`, `astd`, `ajgt`. Columns:

| Column | Meaning |
|---|---|
| `text` | review or tweet, whitespace-normalized, otherwise raw |
| `label` | 0 negative, 1 positive |
| `split` | `train`, `val` or `test` |
| `dataset` | which source |
| `variety` | coarse, per dataset (`mixed`, `egyptian`, `jordanian`) |
| `stars` | original star rating (LABR, HARD), otherwise NaN |
| `group` | hotel name (HARD), otherwise empty |

Run it with `uv run python -m arabic_sentiment.data`.

## Why: the decisions

### 1. One binary label for everything

| Dataset | Rule |
|---|---|
| LABR, HARD | stars 4 and 5 positive, stars 1 and 2 negative, 3 stars dropped (HARD has none) |
| ASTD | keep Negative and Positive, drop Objective and Neutral |
| AJGT | already binary |

Why: every number in the results table is then the same task. Mixing a 3-class score for one dataset with binary scores for the others would make the table misleading. Cost: neutral text is not represented, and a deployed model has to push neutral text into a polarity.

### 2. Store raw text, preprocess later

The only change made to the text is whitespace normalization (`" ".join(s.split())`). No diacritic removal, no letter normalization, no stemming.

Why: different models need different input. The TF-IDF baseline tests normalization on and off (Chapter 7). The transformers come with their own tokenizers. If normalization were baked into the stored data, it could not be undone, and the "normalization on or off" comparison would be impossible.

### 3. Duplicates and conflicting labels

Rule: if a text appears several times with the same label, keep one copy. If it appears with different labels, drop every copy.

Why: duplicates that land in different splits inflate test scores, because the model has seen the answer. A text with two labels is noise whichever you choose, so it is removed.

### 4. Splits

| Dataset | Split | Why |
|---|---|---|
| LABR | official test kept, 10% of train as validation | comparable with published work |
| ASTD, AJGT | seeded stratified 80/10/10 | no official split exists |
| HARD | 80/10/10 by hotel | reviews of one hotel share names and room details |

Validation chooses epochs, configurations and prompts. Test is scored once per final model. Seed 42 everywhere.

## Alternatives

| Choice | Alternative | Why it was not used |
|---|---|---|
| Binary label | Three classes, or five-star regression on reviews | Not comparable across datasets. AJGT has no neutral class. 3-star reviews are ambiguous |
| Raw text stored | Normalize once in the stored data | Irreversible, and it removes the ability to compare preprocessing choices |
| Plain tokenizer input for AraBERT | AraBERT's own preprocessing package | **Not tested.** The model card for AraBERT recommends its own preprocessor (from memory, not checked). All AraBERT runs here used raw text, so its scores may be a little lower than possible |
| Exact-match dedupe | Near-duplicate detection (for example MinHash) | Not done. Near-identical reviews with small edits can still cross splits |
| 80/10/10 split on tiny datasets | k-fold cross-validation | Gives lower-noise estimates on AJGT and ASTD, but multiplies every run, including LLM evaluation. Seeds plus bootstrap intervals were used instead |
| Random row split for HARD | What published HARD work does | Optimistic. Hotel-level split is stricter. Our numbers are not comparable with random-split results |
| Hotel split with plain `GroupShuffleSplit` | `StratifiedGroupKFold`, which also keeps class balance | Not tried. Our val set drifted to 54.6% positive |
| CSV storage | Parquet | Parquet keeps types and handles Arabic text without encoding problems. Needs `pyarrow` |

## How: the core code

### Helper functions

```python
def clean_ws(s):
    return " ".join(str(s).split())
```

`split()` with no argument splits on any run of whitespace (spaces, tabs, newlines), and `" ".join` rejoins with single spaces. `str(s)` guards against non-string values.

```python
def dedupe(df, col):
    n_labels = df.groupby(col).label.transform("nunique")
    return df[n_labels == 1].drop_duplicates(col)
```

- `groupby(col).label.transform("nunique")` gives every row the number of distinct labels among all rows with the same text.
- `df[n_labels == 1]` keeps only texts whose copies all agree, which drops every copy of a conflicting text.
- `.drop_duplicates(col)` then keeps one copy of the rest.

### Stratified split (ASTD, AJGT, and the LABR validation set)

```python
def stratified_split(df):
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.label, random_state=SEED)
    split = pd.Series(index=df.index, dtype=object)
    split[tr.index], split[va.index], split[te.index] = "train", "val", "test"
    return split
```

- The first call takes 20% out as a temporary pool, stratified so the pool has the same class mix as the data.
- The second call halves the pool into validation and test, again stratified. Result: 80/10/10 with the same positive share in each.
- `split[tr.index] = ...` assigns by index label, so the dataframe's index must be unique. That is why the loaders call `.reset_index(drop=True)` after deduping.

### Group split (HARD)

```python
def group_split(df, group_col):
    tr_idx, tmp_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df[group_col]))
    tmp = df.iloc[tmp_idx]
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(tmp, groups=tmp[group_col]))
    split = pd.Series("train", index=df.index)
    split.iloc[tmp_idx[va_rel]] = "val"
    split.iloc[tmp_idx[te_rel]] = "test"
    return split
```

- `GroupShuffleSplit` splits on whole groups: it assigns about 20% of the hotels to the pool, and every review of those hotels goes with them.
- Row shares are therefore only approximately 80/10/10, and class balance is not controlled.
- `next(...)` takes the single split generated. The second split is applied to the pool, and its indices are relative to the pool, hence `tmp_idx[va_rel]` to map back.

### The HARD descriptor rule

```python
QUOTES = " “”\""
DESCRIPTORS = {"استثنائي", "ممتاز", "رائع", "جيد جداً", "جيد",
               "مقبول", "ضعيف", "ضعيف جداً", "مخيب للأمل"}

def lead(text):
    return text.partition(".")[0].strip(QUOTES)

def strip_descriptor(text):
    head, _, rest = text.partition(".")
    if head.strip(QUOTES) in DESCRIPTORS:
        return rest.lstrip(" .")
    return text
```

- `text.partition(".")` returns `(before first period, ".", after)`. HARD reviews start with a short segment ended by a period.
- The head is compared against nine exact phrases after stripping spaces and quote marks.
- If it matches, the text after the first period is returned, without leading spaces or periods. If it does not match, the text is unchanged.
- The nine phrases came from a table of first segments against star rating (Chapter 5).

### Loading HARD

```python
df = pd.read_csv(HARD_FILE, sep="\t", encoding="utf-16", quoting=3)
df = df.rename(columns={"Hotel name": "group", "review": "text", "rating": "stars"})
df["text"] = df.text.map(clean_ws)
df["label"] = (df.stars >= 4).astype(int)
head = df.text.map(lead)
counts = head[head.isin(DESCRIPTORS)].value_counts()
missing = DESCRIPTORS - set(counts.index)
assert not missing, f"descriptors never matched: {missing}"
df["text_notitle"] = df.text.map(strip_descriptor)
df = df[df.text_notitle.str.len() >= 3]
df = dedupe(dedupe(df, "text"), "text_notitle").reset_index(drop=True)
df["split"] = group_split(df, "group")
```

- The `assert` catches a typo in the phrase list: a phrase that never matches would silently do nothing.
- Rows whose stripped text is shorter than 3 characters are dropped (reviews that consisted only of the descriptor).
- The double `dedupe` makes both variants use identical rows. The first removes duplicate and conflicting raw texts, the second removes texts that collide only after stripping. Without it, `hard` and `hard_notitle` would differ in row sets, and the comparison between them would not be like for like.

### Loading LABR

```python
raw = load_dataset("mohamedadaly/labr")
names = raw["train"].features["label"].names          # ['1'..'5'], index = stars - 1
for s in ("train", "test"):
    d = raw[s].to_pandas()
    d["stars"] = d.label.map(lambda i: int(names[i]))
    d = d[d.stars != 3].copy()
    d["label"] = (d.stars >= 4).astype(int)
    d["text"] = d.text.map(clean_ws)
    parts[s] = dedupe(d, "text")
tr, te = parts["train"], parts["test"]
te = te[~te.text.isin(set(tr.text))]                  # leakage fix
tr, va = train_test_split(tr, test_size=0.1, stratify=tr.label, random_state=SEED)
```

- The class label stored on the hub is an index, not the star value. The mapping through `names` avoids an off-by-one error that would flip results silently.
- The leakage line removes any test text that also appears in train.

### Verification

```python
t = [set(df[df.split == s].text) for s in ("train", "val", "test")]
assert not (t[0] & t[1] or t[0] & t[2] or t[1] & t[2])
if df.group.notna().any():
    g = [set(df[df.split == s].group) for s in ("train", "val", "test")]
    assert not (g[0] & g[1] or g[0] & g[2] or g[1] & g[2])
```

Every build asserts no text appears in two splits and, for HARD, no hotel appears in two splits. A leak therefore stops the script instead of passing quietly.

## What went wrong, and what is still weak

| Item | Detail |
|---|---|
| HARD file | UTF-16 with a BOM. Reading it as UTF-8 would have given garbage without an error |
| Size drop larger than predicted | I predicted about 1,662 removed rows. The build removed 2,664 (105,698 to 103,034), because stripping the descriptor creates new duplicates and short remainders. I did not break the 2,664 down by cause |
| LABR leak numbers | I first reported 65 texts shared by train and test. After dropping 3-star rows and deduping, the build printed 2,338 test rows reduced to 2,283, so 55 |
| Package not importable | The build-system block in `pyproject.toml` was duplicated and had to be rewritten (Chapter 2). Separately, a pasted heredoc never created `data.py` once |
| **The descriptor rule is blunt** | The matcher removes quote marks before comparing, so a quoted user title equal to a descriptor word is stripped like an auto label. Example: row 0 of the raw file, a 2-star review starting `“ممتاز”`. Some of the 46,990 matched rows are user titles |
| **Unmatched rows keep their leading segment** | About 56% of rows are not matched by the nine phrases and keep whatever starts them, which may still carry sentiment. `hard_notitle` is therefore not a guaranteed clean measure, and I should not have called it a pure lower bound |
| **LABR is split by review, not by book** | The hub copy has no book id. Reviews of the same book can sit in train and test, which can inflate scores. Not measured |
| `variety` is coarse | ASTD is labeled `egyptian`, but a dialect classifier tagged about half of its test tweets as MSA (Chapter 13). The column is never used as a feature |
| Near-duplicates | Only exact duplicates are removed |

## Numbers

Measured by the build:

| Dataset | Source rows | After processing | Train / val / test | Positive share (train / val / test) |
|---|---|---|---|---|
| LABR | 14,695 official | 8,301 + 923 + 2,283 | 8,301 / 923 / 2,283 | 0.504 / 0.504 / 0.505 |
| HARD | 105,698 | 103,034 | 81,906 / 10,421 / 10,707 | 0.491 / 0.546 / 0.493 |
| ASTD | 9,694 (2,419 binary) | 2,419 | 1,935 / 242 / 242 | 0.321 / 0.322 / 0.318 |
| AJGT | 1,800 | 1,800 | 1,440 / 180 / 180 | 0.500 / 0.500 / 0.500 |

LABR test: 2,338 rows reduced to 2,283 by removing texts present in train.

HARD descriptor matches (rows whose first segment equals a listed phrase): مخيب للأمل 9,535; جيد 8,197; استثنائي 7,311; ضعيف 6,546; ضعيف جداً 5,181; ممتاز 3,970; رائع 3,218; جيد جداً 2,661; مقبول 371. Sum 46,990, about 44% of 105,698 (my addition of the printed counts).

Why the HARD validation share is 54.6%: whole hotels land in one split, and hotel mix differs, so class balance drifts. Macro-F1 is insensitive to this, and the effect is real, not a bug.

## Check yourself

1. Why does `dedupe` drop every copy of a conflicting text but keep one copy of an agreeing one?
2. What would silently go wrong if LABR's label index were used as the star value?
3. Why is `reset_index(drop=True)` needed before `stratified_split`?
4. Why does HARD get a hotel-level split, and what are two side effects of it?
5. Why is the HARD text deduped twice, and what breaks if only once?
6. Name two ways the descriptor rule can be wrong.
7. LABR is split by review. What could that do to its scores, and how would you test it?
8. Why is raw text stored, and what did that cost for AraBERT?
