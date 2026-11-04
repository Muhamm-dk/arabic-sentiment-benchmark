"""Chance-corrected agreement of the two dialect classifiers on rows the main one calls non-MSA.
Run: uv run python -m arabic_sentiment.dialect_kappa
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from arabic_sentiment.dialect_analysis import MIN_CONF, ROOT

MAP = {"GLF": "Gulf", "EGY": "Egyptian", "LEV": "Levantine", "MAGHREB": "Maghrebi"}


def main():
    t = pd.read_parquet(ROOT / "results/dialect_tags.parquet")
    for d, g in t.groupby("dataset"):
        g = g[(g.conf >= MIN_CONF) & (g.dialect != "MSA")]
        a, b = g.dialect.map(MAP), g.dialect_qadi
        pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        chance = sum(pa.get(c, 0) * pb.get(c, 0) for c in set(pa.index) | set(pb.index))
        print(f"{d}: n={len(g)} | agreement {(a == b).mean():.1%} | expected by chance {chance:.1%}"
              f" | kappa {cohen_kappa_score(a, b):.2f}")
        print("   main  :", pa.round(2).to_dict())
        print("   second:", pb.round(2).to_dict())


if __name__ == "__main__":
    main()
