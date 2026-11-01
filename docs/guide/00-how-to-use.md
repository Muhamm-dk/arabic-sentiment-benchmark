# Chapter 0: How to use this guide

## What this guide is

A study guide for one project: a benchmark of Arabic sentiment analysis, built from an empty folder to a deployed API. Every chapter explains what was done, why, what else was possible, how it was implemented, what broke, and what the numbers say. It is written for the author to study from, not as a tutorial for strangers.

Two kinds of statements appear throughout, and the guide keeps them apart:

- **Measured here:** numbers and behaviors produced by our own runs (for example, "TF-IDF scored 0.9411 on HARD").
- **General knowledge:** claims about other datasets, models or tools that were not run in this project. These are marked, and anything not checked is flagged as unverified.

If a sentence in a chapter says "I did not test this", take it literally.

## Chapter template

Each chapter uses the same sections:

1. **What**: the step, in a few lines.
2. **Why**: the decision and the evidence behind it.
3. **Alternatives**: what else was possible and why it lost.
4. **How**: the core code, explained piece by piece. Boilerplate stays in the repo.
5. **What went wrong**: real bugs and detours.
6. **Numbers**: results from our runs.
7. **Check yourself**: questions an interviewer could ask.

## Repo map

```
arabic-sentiment-benchmark/
  pyproject.toml, uv.lock        environment (uv), pinned dependencies
  Dockerfile, .dockerignore      CPU inference image
  run_stage1.sh .. run_stage3.sh, run_xlmr_tuned.sh   training runners
  run_llm_queue.sh               resumable LLM evaluation runner
  src/arabic_sentiment/
    data.py                      build unified, deduplicated, split datasets
    baseline.py, baseline_preds.py   TF-IDF + logistic regression
    finetune.py                  encoder fine-tuning with MLflow
    llm_eval.py                  few-shot LLM evaluation (Groq)
    compare.py, report.py        results tables
    dialect.py, dialect_analysis.py  dialect tagging and per-dialect scores
    combined.py, combined_report.py  the combined-data model
    serve/app.py                 FastAPI service
  tests/                         API and auth tests
  docs/                          datasets.md, error_analysis.md, interview_notes.md, guide/
  results/                       score files (.json, .csv); prediction files are not tracked
  data/                          raw and processed data (not tracked)
```

Files that contain dataset texts (`results/preds/`, `results/llm/*.jsonl`, `results/dialect_tags.parquet`, everything under `data/`) are not tracked in git because the datasets do not state licenses. Scripts that read them need the pipeline rerun first.

## Glossary

| Term | Meaning in this project |
|---|---|
| **Macro-F1** | Mean of the F1 score of each class. With two classes, it weighs negative and positive equally, so ignoring the minority class is punished. |
| **MSA** | Modern Standard Arabic, the formal written variety. |
| **Dialect** | Regional spoken and informal written varieties (Egyptian, Jordanian, Gulf and others). |
| **Encoder** | A BERT-style transformer that turns text into vectors. Here fine-tuned with a classification head. |
| **Fine-tuning** | Continuing training of a pretrained model on a labeled task. |
| **Few-shot** | Giving an LLM a handful of labeled examples in the prompt, without training. |
| **TF-IDF** | A text representation that weights terms by how rare they are across documents. |
| **Bootstrap CI** | A confidence interval from resampling test predictions with replacement (1,000 or 500 resamples here). |
| **Seed** | The random number generator setting. Different seeds give slightly different trained models. |
| **Leakage** | Information in the input that gives away the label, or the same example appearing in train and test. |
| **Descriptor** | The leading Booking.com score label in HARD reviews (Chapter 5). |
| **VRAM** | GPU memory. 8 GB on this laptop, about 6.5 GiB usable. |
| **bf16** | A 16-bit number format with the range of 32-bit floats, used for training. |
| **Invalid rate** | Share of LLM replies that were not a usable label. Counted as errors. |

## How to read the numbers

- Macro-F1 values are on a 0 to 1 scale.
- ASTD (242 test rows) and AJGT (180) are small. A bootstrap interval there is about plus or minus 4 to 5 points, so smaller gaps are noise.
- HARD and LABR encoder runs are single seed, and the LLM comparison on those two uses 300-row samples. Those 300-row numbers are not comparable with full-test numbers.
- "Tuned" XLM-R means one fixed alternative configuration, not a search.

## Check yourself

1. Why does this guide separate "measured here" from "general knowledge"?
2. Why are prediction files not committed to git?
3. Why can you not rank two models on ASTD if they differ by 2 points?
