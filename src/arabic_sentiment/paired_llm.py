import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "results/preds"
DATASETS = ["ajgt", "astd", "labr", "hard", "hard_notitle"]
LLMS = ["qwen", "gptoss", "allam"]
BASES = ["tfidf", "marbert"]
N_BOOT, SEED = 2000, 42


def f1_class(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def macro_f1(y, p):
    pos = f1_class(np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 1)), np.sum((y == 1) & (p != 1)))
    neg = f1_class(np.sum((y == 0) & (p == 0)), np.sum((y == 1) & (p == 0)), np.sum((y == 0) & (p != 0)))
    return (pos + neg) / 2


def find(d, m, s):
    pat = re.compile(rf"{d}-{m}-s{s}(-n\d+)?\.parquet")
    hits = [f for f in PREDS.glob(f"{d}-{m}-s{s}*.parquet") if pat.fullmatch(f.name)]
    return hits[0] if hits else None


def llm_rows(model, d):
    f = ROOT / f"results/llm/{model}-{d}-test-en.jsonl"
    r = pd.DataFrame([json.loads(line) for line in open(f, encoding="utf-8")]).drop_duplicates("i")
    return r[["text", "gold", "pred"]].rename(columns={"gold": "label", "pred": "pred_llm"})


def main():
    rows = []
    for d in DATASETS:
        for llm in LLMS:
            L = llm_rows(llm, d)
            for base in BASES:
                res = {}
                for s in ((0,) if base == "tfidf" else (0, 1, 2)):
                    f = find(d, base, s)
                    if f is None:
                        continue
                    B = pd.read_parquet(f)[["text", "pred"]].rename(columns={"pred": "pred_base"})
                    m = L.merge(B, on="text")
                    assert len(m) == len(L), f"{d} {llm} vs {base} seed {s}: matched {len(m)} of {len(L)} rows"
                    y, pl, pb = m.label.to_numpy(), m.pred_llm.to_numpy(), m.pred_base.to_numpy()
                    res[s] = (macro_f1(y, pl) - macro_f1(y, pb), y, pl, pb)
                if 0 not in res:
                    continue
                diff0, y, pl, pb = res[0]
                rng = np.random.default_rng(SEED)
                boots = np.empty(N_BOOT)
                for k in range(N_BOOT):
                    i = rng.integers(0, len(y), len(y))
                    boots[k] = macro_f1(y[i], pl[i]) - macro_f1(y[i], pb[i])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({"dataset": d, "llm": llm, "base": base, "n": len(y), "invalid": int((pl == -1).sum()),
                             "diff_s0": round(diff0, 4), "ci95": f"[{lo:+.4f}, {hi:+.4f}]",
                             "excludes_0": bool(lo > 0 or hi < 0),
                             "diff_s1": round(res[1][0], 4) if 1 in res else None,
                             "diff_s2": round(res[2][0], 4) if 2 in res else None})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/paired_llm.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
