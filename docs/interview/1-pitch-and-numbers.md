# Interview notes, part 1: pitch, numbers, ground rules

Part 2: concepts. Part 3: the project in depth. Part 4: engineering, scenarios, behavioral questions, weaknesses, live coding.

## The answer shape: claim, evidence, caveat (30 seconds)

1. Claim: one sentence.
2. Evidence: one number and the test set it comes from.
3. Caveat: one limit.

Example: "MARBERT is my best model. On ASTD it scores 0.890 against 0.771 for TF-IDF, over three seeds. Its lead over AraBERT is supported only on LABR."

- Not known: "I did not test that. Here is how I would." Never guess a number. Say where it lives (the results folder, a chapter of the guide).
- Say "within noise" for gaps inside the interval. Say "supported" only when a paired interval excludes 0.
- Say "I measured" or "I inferred". The HARD leak being Booking.com's score labels is inferred.

## 30-second pitch

"I built a benchmark of Arabic sentiment analysis on formal Arabic and dialect. On four public datasets I compared a TF-IDF baseline, three fine-tuned transformers and three few-shot LLMs, scored on identical rows with bootstrap intervals. I found a label leak in the HARD hotel dataset that inflates every score by 5 to 10 points, so I report it two ways. The best encoder, MARBERT, trained on all four datasets, runs behind a tested FastAPI service in Docker, with weights on the Hugging Face Hub and a live demo on Azure."

## 2-minute pitch

- **Problem:** Arabic is many varieties: formal MSA and regional dialects, with unstable spelling. Sentiment models need testing on both.
- **Data:** LABR (book reviews), HARD (hotel reviews), ASTD (Egyptian tweets), AJGT (Jordanian tweets). One binary label, duplicates and conflicting labels removed, HARD split by hotel.
- **Methods:** TF-IDF with logistic regression, fine-tuned AraBERT, MARBERT and XLM-R, and Qwen, gpt-oss and ALLaM with eight examples in the prompt. Validation picks, test judges, three seeds on the tiny sets, paired bootstrap on identical rows.
- **Findings:** 1) Fine-tuned Arabic encoders beat TF-IDF everywhere. 2) HARD leaks its labels through a leading score phrase: remove it and every system drops 5 to 10 points. 3) No dialect gap detected between MSA and dialect text, but the dialect labels are predicted and unvalidated. LLMs: only Qwen on ASTD matches the best encoder.
- **Engineering:** one MARBERT trained on all four datasets, FastAPI with validation and a key, 13 tests, a Docker image, weights on the Hub, deployed on Azure Container Apps.
- **Limits:** single seeds on the big datasets, a 20,000-row HARD training subsample, 300-row LLM samples, an AI assistant tagged the error analysis, no Gulf data.

## 5-minute walkthrough, in order

1. **Why:** Arabic dialects and a UAE audience. A benchmark teaches more than one score.
2. **Data:** four datasets, binary label, why. Gap: no Gulf data.
3. **Rules:** same rows for every system, validation selects, test once, uncertainty always.
4. **Baseline:** TF-IDF, character n-grams win. It defines what fine-tuning buys.
5. **Encoders:** three models, same recipe. MARBERT best. XLM-R is config-sensitive.
6. **The HARD leak:** found from a first-word table, measured, two variants.
7. **LLMs:** few-shot, same rows. The quota stall and what it taught.
8. **Dialect and errors:** null result and why it is limited. Most errors are label problems.
9. **Serving and deployment:** combined model, FastAPI, Docker, Hub, Azure. Cold start 27 s.
10. **Limits and next steps.**

## Numbers to know cold

| Item | Value |
|---|---|
| Datasets, train / val / test | LABR 8,301 / 923 / 2,283. HARD 81,906 / 10,421 / 10,707. ASTD 1,935 / 242 / 242. AJGT 1,440 / 180 / 180 |
| Raw sizes | LABR 14,695 official rows (63,257 full). HARD 105,698 (README says 93,700). ASTD 9,694 (2,419 binary, 68% negative). AJGT 1,800 |
| TF-IDF macro-F1 | LABR 0.8155, HARD 0.9411, HARD no-descriptor 0.8710, ASTD 0.7708, AJGT 0.8722 |
| MARBERT | LABR 0.8799, HARD 0.9561, HARD no-descr. 0.9045, ASTD 0.8902, AJGT 0.9277 |
| AraBERT | 0.8629, 0.9568, 0.9067, 0.8665, 0.9110 |
| XLM-R | 0.8480, 0.9530, 0.9009, 0.8367, 0.8685 (tuned: ASTD 0.8067, AJGT 0.9037) |
| Leak gap | TF-IDF -7.0, encoders -5.0 to -5.2, LLMs -5.1 to -8.2 |
| Descriptor rule | 9 phrases, matches 43% of rows (44,407 of 103,034). Strips quoted user titles in 6.0% of rows. Strict rule: TF-IDF +0.56 |
| Encoder seeds | 3 on ASTD and AJGT (std about 0.01, AraBERT on ASTD 0.032), 1 elsewhere |
| LLMs, ASTD | Qwen 0.8966, gpt-oss 0.8265 (7.4% invalid), ALLaM 0.7522 (6.6% invalid) |
| Qwen against MARBERT on ASTD | 0.8966 against MARBERT's 3-seed mean of 0.8902 (+0.6). Paired against seed 0: -0.2 points, interval [-3.5, +3.1]. Against seeds 1 and 2: about +0.5 and +1.7 |
| MARBERT over AraBERT | supported on LABR only: +1.7 [+0.4, +3.1] |
| Combined model | AJGT 0.933, ASTD 0.869, HARD no-descr. 0.908, LABR 0.864. LABR is 1.6 below the specialist [-2.8, -0.4] |
| Training | AdamW, lr 2e-5, weight decay 0.01, 10% warmup, linear decay, batch 32, bf16, clip 1.0, max length 96 tweets and 256 reviews |
| Hardware | RTX 5070 laptop, 8 GB (about 6.6 GiB usable). Peak VRAM: AraBERT 2.9, MARBERT 3.2, XLM-R 5.2 GiB on tweets. XLM-R 6.0 on reviews |
| Quota | 200,000 tokens a day per model (Qwen, gpt-oss). gpt-oss 30 requests a minute |
| Serving | 13 tests. Image 1.8 GB (387 MB compressed). 405 MiB memory after one request. 0 label disagreements in 1,022 rows against saved predictions |
| Azure | Container Apps, UAE North, 1 vCPU, 2 GiB, 0 to 1 replicas. Cold start 27.5 s (one sample). Warm 0.9 s round trip |
| Calibration | Confidence is overconfident: LABR single MARBERT expected calibration error 0.102, rows at 0.9 or more right 89.6% of the time |

## Say this, not that

| Not that | Say this |
|---|---|
| "MARBERT beats AraBERT." | "MARBERT is ahead on LABR by 1.7 points, supported by a paired test. On the other sets they are level." |
| "Encoders beat LLMs." | "MARBERT scored above every LLM on four of five sets. On ASTD Qwen is nominally ahead by 0.6 points, which is noise. LLM samples are 300 rows on LABR and HARD." |
| "XLM-R is worse." | "XLM-R is config-sensitive on small data. My one alternative configuration helped AJGT and hurt ASTD. Neither beats MARBERT." |
| "The HARD labels are Booking.com's." | "Consistent with Booking.com's score categories. I inferred it from the data and did not check the paper." |
| "There is no dialect effect." | "I detected no gap. The dialect labels are predicted, they bias a gap toward zero, and I have no Gulf data." |
| "The confidence is a probability." | "It is a ranking signal. It is overconfident: rows at 0.9 or more were right 89.6% of the time on LABR." |
| "The deployed model is the best." | "It matches the specialists within noise on three datasets and is 1.6 points lower on LABR." |
| "I proved X." | "The data supports X, within these limits." |

## The honest answer about AI assistance

This section must say what is true for you. The record is: an AI assistant (Claude) guided the project step by step. It proposed the plan, wrote most scripts and drafted the documents. You ran every command, read every output, approved or changed each decision, and caught or triggered corrections. The assistant also tagged the 61 errors in the error analysis, which is why that analysis is flagged as indicative.

Suggested answer: "I built it with an AI assistant as a step-by-step guide. I ran everything myself, and the repository records each decision and each correction. Some of the best findings came from checking the assistant's own claims: it predicted XLM-R would improve with more training, and my run showed it got worse on ASTD. The assistant also tagged the 61 errors, so I report that analysis as indicative, and I have a blind sheet to re-tag them myself."

Rules: do not claim to have written code you could not explain. Anything in the repository is fair game. If you cannot explain a part, study its chapter before the interview, or say you are less sure of it. Interviewers value an honest account of tools and verification.

## Rehearsal plan

1. Say the 30-second pitch aloud until it needs no notes. Then the 2-minute one.
2. Read each guide chapter's "Check yourself" list and answer aloud.
3. Re-tag the 61 errors in `data/processed/error_sheet.csv` before reading the assistant's table. It gives you a real personal contribution.
4. Say "quiz" in the chat and I will interview you one question at a time, from basic to hard, and grade the answers against these notes.
5. Reread the numbers table the evening before.

End of part 1.
