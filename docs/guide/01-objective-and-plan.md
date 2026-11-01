# Chapter 1: Objective and plan

## What

Build a benchmark of Arabic sentiment analysis across Modern Standard Arabic and dialects, then turn the best model into a deployed service. The project is a portfolio piece for ML Engineer and AI Engineering roles at UAE employers, so it has to show both research discipline (fair comparisons, honest limits) and engineering (tracking, tests, containers, deployment).

## The original brief

These were the requirements at the start, in the author's words condensed:

- Datasets: LABR, HARD, ASTD, AJGT.
- Models: fine-tuned encoders (AraBERT, MARBERT, XLM-R) versus a few-shot Arabic-capable LLM (Jais or similar).
- A TF-IDF + logistic regression baseline, a results table, and an error analysis focused on dialect failures.
- A FastAPI service, containerized with Docker, and deployed.
- Experiment tracking with MLflow or Weights & Biases.
- A clean GitHub repo with about 12 meaningful commits and a strong README.
- Constraints: a Lenovo LOQ laptop (RTX 5070, 8 GB VRAM, 32 GB RAM, Windows 11 with WSL2), free Kaggle and Colab GPUs as overflow, 3 to 4 weeks.

## Why this project

**Why sentiment analysis?** It is a small, well-understood task, so the effort goes into method (data quality, evaluation, comparison) and not into task definition. Labels exist in public datasets, and the problem is simple enough for one laptop.

**Why Arabic with dialects?** Arabic is a strong fit for UAE employers, and dialect variation is a real engineering problem: one language, many written varieties, and spelling that is not standardized. A benchmark that covers both MSA and dialect is more informative than one that covers either alone.

**Why a benchmark, and not a single model?** A single model with a good score says little. The value is in the comparison: what does a fine-tuned encoder buy over TF-IDF, and how does a prompted LLM compare? And in the discipline around it: same rows, same splits, uncertainty reported.

**Why serve and deploy it?** Employers for ML engineering roles want to see a model leave the notebook. Serving, containerizing, testing and deploying show that.

## Alternatives to this project framing

| Alternative | Why it lost |
|---|---|
| One model, one dataset, one score | Easy, but nothing to defend and nothing to learn from |
| 3-class sentiment (adding neutral) | Cleaner for real use, but AJGT has no neutral class and 3-star reviews are ambiguous, so scores would not be comparable across datasets |
| A larger, newer LLM-only study | Cheaper to run, but free API caps limit test size, and it skips fine-tuning, which the roles value |
| Training from scratch | Pointless at this data size; pretrained models do far better |
| Many datasets (10 or more) | Dilutes the analysis. Four datasets already span two domains and several dialects |

Whether better datasets or models exist is covered where they matter, in Chapters 3 and 8.

## The plan

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data and baseline | Four datasets in one schema, fixed splits, TF-IDF baseline logged |
| 2 | Encoder fine-tuning | AraBERT, MARBERT, XLM-R on all datasets, first full results table |
| 3 | LLM few-shot and error analysis | LLM runs, dialect analysis, error write-up |
| 4 | Serving and polish | FastAPI, Docker, deployment, README with honest limitations |

Principles that governed every step:

1. **Same rows for every system.** Differences between models must come from the models, not from the data they saw.
2. **Validation picks, test judges.** Epochs, prompts and configurations are chosen on the validation split, and test is scored once per final model.
3. **Report uncertainty.** Bootstrap intervals, several seeds where test sets are tiny.
4. **Report what failed.** Leaks, bugs and weak results go in the write-up.
5. **Decide before looking.** Where a choice could be tuned toward a good-looking result (the prompt language, the tuned XLM-R configuration), the rule was fixed in advance.

## What actually happened versus the plan

- **Jais was replaced by ALLaM.** Jais 13B does not fit on the laptop, and the Kaggle route needed custom model code and 4-bit loading that I could not verify would work. ALLaM, an Arabic-specialist model, was available on the Groq API, so it filled the same role with much less risk. The Kaggle option was dropped.
- **Two extra LLMs.** Qwen and gpt-oss ran alongside ALLaM, giving a general-purpose comparison.
- **The LLM phase took days, not hours.** Free-tier caps of 200,000 tokens per day per model turned a one-evening job into several days of resumed runs (Chapter 11).
- **A data leak changed the evaluation.** HARD's leading score label inflated every score, so HARD became two benchmark variants (Chapter 5).
- **Deployment moved clouds.** Hugging Face Docker Spaces were not selectable on a free account, so the service runs on Azure Container Apps (Chapter 18).
- **Commit count.** The repo has more than the planned 12 commits, because the project grew.

## Numbers: the project in one table

| Dataset | TF-IDF | Best encoder | Notes |
|---|---|---|---|
| LABR | 0.8155 | MARBERT 0.8799 | single seed |
| HARD (as released) | 0.9411 | AraBERT 0.9568 | inflated by the leak |
| HARD (descriptor removed) | 0.8710 | AraBERT 0.9067 | the honest HARD number |
| ASTD | 0.7708 | MARBERT 0.8902 | 3-seed mean, small test set |
| AJGT | 0.8722 | MARBERT 0.9277 | 3-seed mean, small test set |

Macro-F1 on the full test splits. HARD encoder runs used a 20,000-review training subsample.

## What success looks like

Not "highest score", but:

- Every number can be reproduced from the repo.
- Every comparison is on identical rows and every gap is judged against its noise.
- Every limitation is written down.
- The deployed service works, is tested, and is protected by a key.

## Check yourself

1. Why was the project framed as a benchmark, and not as one model with one score?
2. Why was binary sentiment chosen over three classes?
3. What does "validation picks, test judges" prevent?
4. Why was ALLaM used in place of Jais, and what was lost?
5. Name two decisions that were fixed before looking at results, and why that matters.
