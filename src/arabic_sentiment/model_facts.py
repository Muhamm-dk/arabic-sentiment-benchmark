"""Model and tokenizer facts for Chapter 8. Run: uv run python -m arabic_sentiment.model_facts"""
import pandas as pd
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "marbert": "UBC-NLP/MARBERT",
    "xlmr": "FacebookAI/xlm-roberta-base",
}
SENTENCE = "الخدمة كانت سيئة جدا ومش هرجع تاني"


def main():
    samples = []
    for d in ["labr", "hard_notitle", "astd", "ajgt"]:
        t = pd.read_parquet(f"data/processed/{d}.parquet")
        t = t[t.split == "test"].text
        samples.append((d, t.sample(min(300, len(t)), random_state=0).tolist()))
    for key, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        n = sum(p.numel() for p in model.parameters())
        emb = model.get_input_embeddings().weight.numel()
        print(f"\n{key} ({name})")
        print(f"  vocab {len(tok)} | params {n / 1e6:.1f}M | embeddings {emb / 1e6:.1f}M ({emb / n:.0%})"
              f" | fixed training memory {n * 16 / 2**30:.2f} GiB")
        print("  pieces:", tok.tokenize(SENTENCE))
        for d, texts in samples:
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            words = sum(len(t.split()) for t in texts)
            toks = sum(len(i) for i in ids)
            unk = sum(i.count(tok.unk_token_id) for i in ids) if tok.unk_token_id is not None else 0
            print(f"  {d:13s} tokens per word {toks / words:.2f} | unknown-piece share {unk / toks:.4%}")


if __name__ == "__main__":
    main()
