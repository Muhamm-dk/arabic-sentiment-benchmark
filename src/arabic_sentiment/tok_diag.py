"""Tokenizer diagnostics for Chapter 8. Run: uv run python -m arabic_sentiment.tok_diag"""
import collections

import pandas as pd
from transformers import AutoTokenizer

MODELS = {"arabert": "aubmindlab/bert-base-arabertv02", "marbert": "UBC-NLP/MARBERT"}


def main():
    labr = pd.read_parquet("data/processed/labr.parquet").query("split == 'test'").text.tolist()
    astd = pd.read_parquet("data/processed/astd.parquet").text.tolist()
    for key, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        print(f"\n== {key} | fast tokenizer: {tok.is_fast}")
        keep = ("do_lower_case", "strip_accents", "tokenize_chinese_chars")
        print("settings:", {k: v for k, v in tok.init_kwargs.items() if k in keep})
        print("hamza test:", tok.tokenize("سيئة أكل إلى مؤسسة آمن"))
        n = pd.Series([len(x) for x in tok(labr)["input_ids"]])
        print("LABR test tokens p50/p90/p99:", int(n.quantile(.5)), int(n.quantile(.9)), int(n.quantile(.99)),
              "| share > 128:", round((n > 128).mean(), 3), "| share > 256:", round((n > 256).mean(), 3))
        if not tok.is_fast:
            continue
        for label, texts in (("astd", astd), ("labr", labr)):
            c, total = collections.Counter(), 0
            for t in texts:
                enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
                for i, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
                    total += 1
                    if i == tok.unk_token_id:
                        c[t[s:e]] += 1
            print(f"{label}: unknown pieces {sum(c.values())} of {total} tokens | most common source text:",
                  c.most_common(15))


if __name__ == "__main__":
    main()
