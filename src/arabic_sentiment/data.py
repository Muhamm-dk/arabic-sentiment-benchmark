"""Build unified, deduplicated, split datasets.

Run: uv run python -m arabic_sentiment.data
Output: data/processed/{labr,hard,hard_notitle,astd,ajgt}.parquet
Columns: text, label (0 neg, 1 pos), split, dataset, variety, stars, group
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
HARD_FILE = ROOT / "data/raw/hard_extracted/balanced-reviews.txt"
OUT = ROOT / "data/processed"
COLUMNS = ["text", "label", "split", "dataset", "variety", "stars", "group"]
QUOTES = " “”\""

# Leading score labels in HARD reviews, chosen from the segment-by-rating crosstab (docs/datasets.md)
DESCRIPTORS = {
    "استثنائي", "ممتاز", "رائع", "جيد جداً", "جيد",
    "مقبول", "ضعيف", "ضعيف جداً", "مخيب للأمل",
}


def clean_ws(s):
    return " ".join(str(s).split())


def dedupe(df, col):
    """Drop every copy of a text that has conflicting labels, keep one copy of the rest."""
    n_labels = df.groupby(col).label.transform("nunique")
    return df[n_labels == 1].drop_duplicates(col)


def lead(text):
    return text.partition(".")[0].strip(QUOTES)


def strip_descriptor(text):
    head, _, rest = text.partition(".")
    if head.strip(QUOTES) in DESCRIPTORS:
        return rest.lstrip(" .")
    return text


def stratified_split(df):
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.label, random_state=SEED)
    split = pd.Series(index=df.index, dtype=object)
    split[tr.index], split[va.index], split[te.index] = "train", "val", "test"
    return split


def group_split(df, group_col):
    tr_idx, tmp_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df[group_col]))
    tmp = df.iloc[tmp_idx]
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(tmp, groups=tmp[group_col]))
    split = pd.Series("train", index=df.index)
    split.iloc[tmp_idx[va_rel]] = "val"
    split.iloc[tmp_idx[te_rel]] = "test"
    return split


def load_labr():
    raw = load_dataset("mohamedadaly/labr")
    names = raw["train"].features["label"].names  # ['1'..'5'], index = stars - 1
    parts = {}
    for s in ("train", "test"):
        d = raw[s].to_pandas()
        d["stars"] = d.label.map(lambda i: int(names[i]))
        d = d[d.stars != 3].copy()
        d["label"] = (d.stars >= 4).astype(int)
        d["text"] = d.text.map(clean_ws)
        parts[s] = dedupe(d, "text")
    tr, te = parts["train"], parts["test"]
    n_te = len(te)
    te = te[~te.text.isin(set(tr.text))]
    print(f"LABR: test {n_te} -> {len(te)} after removing texts present in train")
    tr, va = train_test_split(tr, test_size=0.1, stratify=tr.label, random_state=SEED)
    out = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    return out.reset_index(drop=True)


def load_hard():
    df = pd.read_csv(HARD_FILE, sep="\t", encoding="utf-16", quoting=3)
    df = df.rename(columns={"Hotel name": "group", "review": "text", "rating": "stars"})
    df["text"] = df.text.map(clean_ws)
    df["label"] = (df.stars >= 4).astype(int)
    head = df.text.map(lead)
    counts = head[head.isin(DESCRIPTORS)].value_counts()
    print("HARD descriptor matches:\n" + counts.to_string())
    missing = DESCRIPTORS - set(counts.index)
    assert not missing, f"descriptors never matched: {missing}"
    df["text_notitle"] = df.text.map(strip_descriptor)
    n0 = len(df)
    df = df[df.text_notitle.str.len() >= 3]
    df = dedupe(dedupe(df, "text"), "text_notitle").reset_index(drop=True)
    print(f"HARD: {n0} rows -> {len(df)} after dedupe and conflict removal")
    df["split"] = group_split(df, "group")
    return df


def load_astd():
    raw = load_dataset("arbml/ASTD")["train"]
    names = raw.features["label"].names
    d = raw.to_pandas()
    d["cls"] = d.label.map(lambda i: names[i])
    d = d[d.cls.isin(["Positive", "Negative"])].copy()
    d["label"] = (d.cls == "Positive").astype(int)
    d["text"] = d.tweet.map(clean_ws)
    n0 = len(d)
    d = dedupe(d, "text").reset_index(drop=True)
    print(f"ASTD: {n0} binary rows -> {len(d)} after dedupe")
    d["split"] = stratified_split(d)
    return d


def load_ajgt():
    raw = load_dataset("komari6/ajgt_twitter_ar")["train"]
    assert raw.features["label"].names == ["Negative", "Positive"]
    d = raw.to_pandas()
    d["text"] = d.text.map(clean_ws)
    n0 = len(d)
    d = dedupe(d, "text").reset_index(drop=True)
    print(f"AJGT: {n0} rows -> {len(d)} after dedupe")
    d["split"] = stratified_split(d)
    return d


def finalize(df, dataset, variety):
    df = df.copy()
    df["dataset"], df["variety"] = dataset, variety
    if "stars" not in df:
        df["stars"] = float("nan")
    if "group" not in df:
        df["group"] = None
    return df[COLUMNS]


def verify(df, name):
    t = [set(df[df.split == s].text) for s in ("train", "val", "test")]
    assert not (t[0] & t[1] or t[0] & t[2] or t[1] & t[2]), f"{name}: text overlap between splits"
    if df.group.notna().any():
        g = [set(df[df.split == s].group) for s in ("train", "val", "test")]
        assert not (g[0] & g[1] or g[0] & g[2] or g[1] & g[2]), f"{name}: group overlap between splits"


def summary(df, name):
    t = df.groupby("split").label.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "pos_rate"})
    print(f"\n{name}\n" + t.round(3).reindex(["train", "val", "test"]).to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hard = load_hard()
    sets = {
        "labr": finalize(load_labr(), "labr", "mixed"),
        "hard": finalize(hard, "hard", "mixed"),
        "hard_notitle": finalize(hard.assign(text=hard.text_notitle), "hard_notitle", "mixed"),
        "astd": finalize(load_astd(), "astd", "egyptian"),
        "ajgt": finalize(load_ajgt(), "ajgt", "jordanian"),
    }
    for name, df in sets.items():
        verify(df, name)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        summary(df, name)


if __name__ == "__main__":
    main()
