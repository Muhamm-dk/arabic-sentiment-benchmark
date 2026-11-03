# Chapter 8: Transformer foundations

## What

What happens between a string of Arabic text and a class label, and what the three encoders we fine-tuned (AraBERT, MARBERT, XLM-R) actually are. This chapter is concepts and facts. The training loop is Chapter 9.

Claim types: **checked** means confirmed against a model card, paper or search result during this project. **Arithmetic** means derived from numbers we have. **Memory** means from general knowledge and not re-checked.

## The pipeline

```
text -> tokenizer -> token ids -> embeddings (+ positions) -> 12 transformer layers -> vector for the first token -> linear layer -> 2 scores -> softmax
```

In `finetune.py`: `encode(tok, texts, max_len)` does the tokenization (with truncation), `tok.pad` pads each batch to its longest item, `model(**x).logits` runs everything else, and `cross_entropy` compares the two scores with the label.

### 1. Tokenization

A model has a fixed vocabulary of subword pieces. Text is cut into pieces from that vocabulary, and each piece has an id. Rare words become several pieces, so any string can be represented.

- **WordPiece** (AraBERT, MARBERT): greedy longest-match against a learned list. Continuation pieces are marked with `##`.
- **SentencePiece** (XLM-R): works on raw text including spaces, with one shared vocabulary for all languages.

Why it matters for Arabic: attached prefixes and suffixes and dialect spellings produce many rare word forms. A vocabulary with more Arabic pieces cuts the text into fewer, more meaningful chunks.

Arabic share of each vocabulary (checked, from the ARBERT and MARBERT paper): mBERT 5K of 110K pieces, XLM-R 14K of 250K, AraBERT 60K of 64K (that figure is for the first AraBERT version), ARBERT and MARBERT 100K of 100K. So XLM-R spends 5.6% of its vocabulary on Arabic. The consequence I expect is more tokens per Arabic word. The script below measures it.

### 2. Embeddings

Each token id selects a row of a learned table of 768 numbers. A position embedding (a second learned table, one row per position, up to 512) is added so the model knows word order. BERT-style models also add a segment embedding, which we never use, since the input is one segment.

### 3. Self-attention

Every token builds a new vector as a weighted mix of all tokens in the sentence. The weights come from comparing tokens with each other:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

- Q, K and V are three linear projections of the same input (queries, keys, values).
- `Q K^T` scores every token pair. The division by `sqrt(d_k)` keeps the scores from growing with dimension. Softmax turns each row into weights that sum to 1.
- **Multi-head:** 12 heads each work in a 64-dimensional slice (768 / 12) and their outputs are concatenated. Different heads can attend to different relations.
- The score matrix is `length x length` per head, so memory grows with the square of the sequence length. That is one reason max length 96 for tweets and 256 for reviews were set deliberately.
- **The padding mask** (`attention_mask` in the code) tells attention to ignore padded positions.

Attention is bidirectional in an encoder: each token sees tokens on both sides. That differs from a generator such as an LLM, which only sees the left context.

### 4. The rest of a layer

After attention, each token passes through a small two-layer network (768 to 3072 to 768, memory of the BERT recipe), and both parts are wrapped in a residual connection (add the input back) and layer normalization. Twelve such layers are stacked. The vector of the first token (`[CLS]` or `<s>`), which attention lets summarize the whole sentence, goes to the classification head.

### 5. Pretraining: masked language modeling

Before we touch it, each encoder was trained on unlabeled text to predict hidden tokens: a share of tokens (15% in the BERT recipe, memory) is hidden, and the model must recover them from context. That forces it to learn vocabulary, syntax and some meaning from a huge corpus. BERT also trained a next-sentence prediction task. MARBERT dropped it because tweets are short (checked). XLM-R also uses only masked language modeling (checked, from secondary summaries).

### 6. Fine-tuning

We attach a new randomly initialized linear layer (and, per the library, a pooler and dropout in front of it; memory) and train the whole network on the labels with cross-entropy. The load report you saw for MARBERT confirms the head: `classifier.weight` and `classifier.bias` were MISSING (newly created), and the `cls.predictions.*` and `cls.seq_relationship.*` weights were UNEXPECTED (pretraining heads, discarded).

## The three models

| | AraBERT v0.2 base | MARBERT | XLM-R base |
|---|---|---|---|
| Hub id | `aubmindlab/bert-base-arabertv02` | `UBC-NLP/MARBERT` | `FacebookAI/xlm-roberta-base` |
| Pretraining data (checked) | 77 GB, 200M sentences, 8.6B words. Sources on the card include Arabic Wikipedia (2020 dump), the 1.5B-words Arabic corpus, OSIAN and Assafir news | 1B tweets sampled from about 6B in-house, 128 GB, 15.6B tokens, MSA and dialects | CC-100, 100 languages, 2.5 TB. Arabic part: 2,869M tokens, 28.0 GiB |
| Tokenizer (checked) | WordPiece, new vocabulary learned for v2 with the `tokenizers` library. v0.2 is not pre-segmented with Farasa | WordPiece, 100K | SentencePiece, 250K, shared |
| Architecture | BERT-base, 12 layers, 768 hidden, 12 heads | BERT-base without next-sentence prediction | RoBERTa-style, 12 / 768 / 12, masked language modeling only |
| Parameters | 136M (card), 135M (paper) | 163M | 278M by the library count (270M in the paper) |
| Pretraining run (checked) | TPUv3-8, 3M steps, lengths 128 and 512 | batch 256, length 128, 14 epochs, 14 days on 8 Cloud TPUs (earlier MARBERT paper version) | not collected |

Preprocessing that AraBERT expects (checked, from its card and package): it recommends applying its `ArabertPreprocessor` before training and testing. For the v0.2 models this means cleaning steps such as stripping tatweel, inserting spaces around punctuation and numbers, repeated-character reduction, mapping `/` to `-` (since `/` is missing from the vocabulary) and mapping Hindi digits. Farasa segmentation applies to the v1 and v2 models, not v0.2. We used raw text.

A conflicting source: one secondary paper says MARBERT was also pretrained on Gigaword, OSCAR, OSIAN and Wikipedia. The UBC repository and model card say 1B tweets. I used the primary source and did not resolve the conflict.

## Where the parameters are (arithmetic)

Most of a base model's size is the embedding table, not the transformer layers.

| Part | Count |
|---|---|
| Attention, 4 matrices of 768 x 768 | 2,359,296 per layer |
| Feed-forward, 768 x 3072 and 3072 x 768 | 4,718,592 per layer |
| Biases and layer norms | about 10,000 per layer |
| 12 layers together | 85,054,464 |
| Position embeddings (514 x 768 for XLM-R), embedding norm, pooler and the rest | about 1M |

Body total: 86,042,112 parameters for XLM-R. Check: XLM-R's 250,002 x 768 embedding table is 192,001,536, and 192,001,536 + 86,042,112 = 278,043,648, which is exactly the library count one source reports. So the accounting is right.

Same arithmetic for the others:

| Model | Embeddings | Body | Total | Reported |
|---|---|---|---|---|
| MARBERT (100,000 x 768) | 76.8M | 86.0M | 162.8M | 163M |
| AraBERT v0.2 (assuming a 64,000-piece vocabulary) | 49.2M | 86.0M | 135.2M | 135M in the paper. 136M on the card, which matches adding the masked-language-model head (about 0.66M) |
| XLM-R base | 192.0M | 86.0M | 278.0M | 278M |

I did not check AraBERT's vocabulary size directly. 64,000 is consistent with both published totals.

So XLM-R has the same transformer body as the other two, and the extra 100 million parameters sit in its 250K-piece embedding table.

## Memory: why this matters on 8 GB (arithmetic, checked against our runs)

Training in fp32 weights with AdamW keeps four numbers per parameter (weight, gradient, two optimizer states), 16 bytes in total. In `finetune.py`, bf16 autocast changes the math but the weights stay fp32.

| Model | Parameters | Fixed memory (16 bytes each) | Measured peak, tweets, length 96, batch 32 | Remainder (activations and overhead) |
|---|---|---|---|---|
| AraBERT | 135.2M | 2.02 GiB | 2.90 GiB | 0.89 GiB |
| MARBERT | 162.8M | 2.43 GiB | 3.21 GiB | 0.78 GiB |
| XLM-R | 278.0M | 4.14 GiB | 5.23 GiB | 1.09 GiB |

The remainders are similar, so the fixed part explains the differences. On reviews at length 256 the peaks were 4.87 GiB (AraBERT, batch 32), 5.18 GiB (MARBERT, batch 32) and 5.98 to 6.0 GiB (XLM-R, batch 16 with gradient accumulation), against about 6.5 GiB usable.

Large models, by the same arithmetic and untested:
- **XLM-R large** (560M): 8.3 GiB of fixed memory alone. It does not fit on this GPU for full fine-tuning.
- **AraBERT large** (371M, from its card): 5.5 GiB fixed, which leaves almost nothing. It would need a small batch, gradient checkpointing or an 8-bit optimizer, none of which I tried.

## Why these three models

I did not run a model-selection search. The brief named them. They form a sensible design:

- **AraBERT**: Arabic-only, text-oriented pretraining (news, Wikipedia, books-like corpora).
- **MARBERT**: Arabic-only, tweet-oriented pretraining, including dialects.
- **XLM-R**: multilingual baseline, to show what Arabic-specific pretraining adds.

What the results show (Chapter 10): MARBERT was best or tied on 4 of 5 test sets, including the review datasets. That is not what "tweet model for tweets, news model for reviews" would predict, so the simple story does not hold. Possible reasons, untested: MARBERT's pretraining data is larger (15.6B tokens against 8.6B words), or its 100K vocabulary fits Arabic better.

## Alternatives

None were run. Figures are checked from search results.

| Model | Facts | Note |
|---|---|---|
| ARBERT | 61 GB, 6.2B tokens, MSA, 163M, 100K vocabulary | Same group as MARBERT |
| MARBERTv2, ARBERTv2 | MARBERTv2: further pretrained on more data with 512-token sequences for 40 epochs. ARBERTv2: 243 GB MSA | A natural upgrade. Not tried |
| AraBERT v0.2 Twitter (base and large) | v0.2 plus 60M multi-dialect tweets | Closer to our tweet sets |
| CAMeLBERT (MSA, DA, CA, Mix) | 30K vocabulary, Mix has 17.3B tokens | Dialect and classical variants |
| QARiB | 97 GB, 14B tokens, MSA plus 420M tweets | |
| AraELECTRA | 135M, discriminator-style pretraining | |
| mBERT | 110K vocabulary, 5K Arabic pieces | Weaker Arabic coverage |
| Larger variants | AraBERT large 371M, XLM-R large 560M | Memory limits above |
| LLMs | Chapter 11 | |

## How: measure the facts yourself

This script prints each model's vocabulary size, parameter count, embedding share, fixed training memory, and tokens per word on a sample of test texts from each dataset. It runs on CPU and takes a few minutes.

```bash
nano src/arabic_sentiment/model_facts.py
```

```python
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
```

```bash
uv run python -m py_compile src/arabic_sentiment/model_facts.py && uv run python -m arabic_sentiment.model_facts
```

The count includes the pooler, so it is about 0.6M above a model without it. Paste the output and I'll put the real numbers into the next section.

## Numbers

Measured with `model_facts.py` (300 test texts per dataset; all of them for ASTD with 242 and AJGT with 180):

| Model | Vocab | Params | Embedding share | Fixed training memory | Tokens per word: LABR / HARD no-descr. / ASTD / AJGT |
|---|---|---|---|---|---|
| AraBERT | 64,000 | 135.2M | 36% | 2.01 GiB | 1.27 / 1.43 / 1.46 / 1.37 |
| MARBERT | 100,000 | 162.8M | 47% | 2.43 GiB | 1.20 / 1.30 / 1.33 / 1.13 |
| XLM-R | 250,002 | 278.0M | 69% | 4.14 GiB | 1.72 / 1.81 / 2.02 / 1.74 |

- The parameter and memory figures match the accounting above, and AraBERT's vocabulary is 64,000 as assumed.
- The prediction made before running held: XLM-R needs the most tokens per word on every dataset, 39 to 54% more than MARBERT (LABR 1.72 against 1.20, HARD 1.81 against 1.30, ASTD 2.02 against 1.33, AJGT 1.74 against 1.13) and 27 to 38% more than AraBERT. Tokens per word divides the token count by whitespace-separated words, so a hashtag counts as one word.
- Sample sentence: AraBERT cuts `هرجع` and `تاني` into `هر ##جع` and `تان ##ي`, MARBERT keeps both whole, and XLM-R produces 11 pieces for 7 words.

## Tokenizer findings (`tok_diag.py`)

Settings: AraBERT `do_lower_case` False, MARBERT `do_lower_case` True (strip_accents unset), both fast tokenizers.

Hamza test, input `سيئة أكل إلى مؤسسة آمن`:
- AraBERT returns the words unchanged.
- MARBERT returns `سيية اكل الى موسسة امن`: hamza forms are folded (أ إ آ to ا, ئ to ي, ؤ to و) and ة is kept. My reading, from memory of the BERT tokenizer: lowercasing also strips combining marks, which removes hamza marks and diacritics such as tanween. `سيية` is a single vocabulary piece, which suggests the vocabulary was built on text normalized the same way. Not verified.

Unknown pieces (full LABR test set of 2,283 reviews, and all 2,419 binary ASTD tweets):

| Model | Text | Unknown pieces | Most common source texts |
|---|---|---|---|
| AraBERT | LABR test | 3,687 of 174,851 (2.1%) | جداً 169, كثيراً 68, `/` 65, جدًا 36, لـ 33, بـ 26, شيئاً 25 |
| AraBERT | ASTD | 674 of 55,261 (1.2%) | `/` 18, لـ 16, بـ 14, جداً 14, ﻭ 9, ♡ 7, • 7 |
| MARBERT | LABR test | 2,264 of 163,690 (1.4%) | `"` 1,357, `_` 559, لـ 33, بـ 26 |
| MARBERT | ASTD | 4,284 of 50,820 (8.4%) | `#` 2,385, `_` 1,507, `"` 294 |
| XLM-R | all four datasets | 0% (from `model_facts.py`; `tok_diag.py` was not run for it) | |

These shares cover whole sets, so they differ slightly from the 300-row samples in `model_facts.py`.

Reading:
- MARBERT's unknowns are punctuation-like markers (`#`, `_`, the straight quote). The words in a hashtag are still tokenized. The effect on accuracy was not measured. MARBERT is still the best model on ASTD.
- AraBERT's unknowns are content words: very common intensifiers and thanks written with tanween. Those are sentiment-bearing positions, which is a plausible handicap on the review datasets. This matches AraBERT's recommended preprocessing, which our runs did not use (open question 1). The effect was not measured.
- Not checked: whether curly quotes are also unknown for MARBERT. It matters for HARD (Chapter 5), where a quoted user title and an unquoted descriptor differ by their quote marks.

## Length coverage (real tokenizers, full LABR test set)

| | p50 / p90 / p99 | Truncated at 128 | Truncated at 256 |
|---|---|---|---|
| AraBERT | 41 / 170 / 692 | 14.3% | 4.9% |
| MARBERT | 38 / 161 / 685 | 13.1% | 4.6% |

About 5% of reviews are truncated at 256 for the two Arabic models. The README's earlier "about 10%" came from the XLM-R tokenizer on a mixed sample and was too high for them. XLM-R's coverage was not measured with this script.

## Open questions

1. **AraBERT preprocessing.** Its card recommends the preprocessing tool, we used raw text, and the unknown pieces above fall on exactly the kinds of items the tool handles (tatweel and the slash are confirmed on the card). Test: preprocess the text, then rerun AJGT and ASTD with 3 seeds and LABR with 1. Not done. The comparison must keep the original runs separate from the new ones.
2. **MARBERT beyond 128 tokens.** Pretraining used sequences of 128 tokens (earlier MARBERT paper version). Measured: MARBERT on LABR at 128 tokens scored test macro-F1 0.8844 against 0.8799 at 256. Paired bootstrap of the difference: +0.0044, 95% interval [-0.0048, +0.0137]. The two models disagree on 116 of 2,283 reviews (63 where only the 128 run is right, 53 the other way). Peak VRAM 3.61 against 5.18 GiB. One seed each, so training noise and the length effect are not separated. Conclusion: no detectable effect of the length. Whether positions 129 to 256 were trained in pretraining was not established.
3. **The pooler.** MARBERT was trained without next-sentence prediction, which is what normally trains the pooler. The checkpoint still lists `cls.seq_relationship.*` weights. My inference is that the pooler may be untrained. Alternative: average token vectors instead of using the first token. Not tried.
4. **Overlap with pretraining data.** MARBERT's tweets may include ASTD and AJGT tweets. Not checked.

## Check yourself

1. Why are subword vocabularies used, and what happens to a word the vocabulary has never seen?
2. In the attention formula, what do Q, K and V come from, and why divide by `sqrt(d_k)`?
3. Why does attention memory grow with the square of the sequence length?
4. Why does XLM-R base have 278M parameters if its transformer body is the same size as BERT-base?
5. Why is training memory about 16 bytes per parameter with AdamW, and why does it make XLM-R large impossible on 8 GB?
6. What did the `UNEXPECTED` and `MISSING` keys in the MARBERT load report mean?
7. Give two reasons the simple story "MARBERT for tweets, AraBERT for reviews" failed.
8. Which of our results could the AraBERT preprocessing recommendation change?
