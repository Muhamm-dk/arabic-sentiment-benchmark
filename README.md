# Arabic Sentiment Benchmark

Binary sentiment analysis (negative vs positive) on Modern Standard Arabic and dialect text. Compares a TF-IDF + logistic regression baseline, fine-tuned encoders (AraBERT, MARBERT, XLM-R) and few-shot LLMs (Qwen, gpt-oss, ALLaM via Groq) on four public datasets, then serves the best model through a FastAPI service in Docker.

Model on the Hub: [Muhamm-dk/arabic-sentiment-marbert-combined](https://huggingface.co/Muhamm-dk/arabic-sentiment-marbert-combined)

## Key findings

1. **Fine-tuned Arabic encoders beat TF-IDF on every dataset.** AraBERT and MARBERT beat it everywhere, and XLM-R ties it on AJGT. MARBERT has the top score on LABR, ASTD and AJGT, and is within 0.3 points of the top score (AraBERT) on both HARD variants. The largest gain is on tweets (ASTD: 0.890 vs 0.771).
2. **HARD contains a label leak.** Most reviews start with a Booking.com score label (for example "استثنائي", "ضعيف جداً") that maps almost one-to-one to a star rating. Removing it costs every system 5 to 10 macro-F1 points, so raw HARD scores are inflated. Both variants are reported.
3. **No dialect gap detected, with caveats.** Per predicted-dialect scores for the encoders overlap between MSA and non-MSA text on LABR and raw HARD, and a manual read of 61 errors found dialect vocabulary plausible in about 13%. Dialect labels come from an unvalidated third-party classifier (it mislabels short Jordanian tweets), which biases any gap toward zero, and on HARD the Gulf label tracks hotel region. XLM-R on Egyptian tweets is one suggestive exception. The analysis does not cover Gulf or Maghrebi text.

## Datasets

Binary label on all four. Exact duplicates removed, texts with conflicting labels dropped.

| Dataset | Source | Variety | Labels | Train / val / test |
|---|---|---|---|---|
| LABR | Book reviews | Mixed | 1-2 stars negative, 4-5 positive, 3 dropped | 8,301 / 923 / 2,283 |
| HARD | Hotel reviews (balanced file) | MSA and dialect | 1-2 negative, 4-5 positive | 81,906 / 10,421 / 10,707 |
| ASTD | Tweets | Mostly Egyptian | Negative and Positive only (Objective, Neutral dropped) | 1,935 / 242 / 242 |
| AJGT | Tweets | Jordanian | Negative, Positive | 1,440 / 180 / 180 |

Split decisions: LABR keeps its official test set with texts also present in train removed (55 rows). ASTD and AJGT use seeded stratified 80/10/10 splits. HARD is split 80/10/10 **by hotel**, so no hotel appears in two splits. Validation drives epoch and configuration selection, and test is scored once. See `docs/datasets.md`.

## Results

Metric: macro-F1 on the test split. ASTD and AJGT test sets are tiny, so differences under about 4 points are noise.

### Full test sets

Encoders on ASTD and AJGT are 3-seed means. LABR and HARD encoder runs are single seed. HARD encoders trained on a fixed 20,000-review subsample.

| Dataset (n test) | TF-IDF | AraBERT | MARBERT | XLM-R | XLM-R (tuned) |
|---|---|---|---|---|---|
| LABR (2,283) | 0.8155 | 0.8629 | **0.8799** | 0.8480 | - |
| HARD, as released (10,707) | 0.9411 | **0.9568** | 0.9561 | 0.9530 | - |
| HARD, descriptor removed (10,707) | 0.8710 | **0.9067** | 0.9045 | 0.9009 | - |
| ASTD (242) | 0.7708 | 0.8665 | **0.8902** | 0.8367 | 0.8067 |
| AJGT (180) | 0.8722 | 0.9110 | **0.9277** | 0.8685 | 0.9037 |

TF-IDF 95% bootstrap CIs: LABR [0.799, 0.830], HARD [0.937, 0.945], HARD no-descriptor [0.865, 0.877], ASTD [0.710, 0.828], AJGT [0.822, 0.917].

"Tuned" XLM-R (lr 3e-5, 10 epochs) was fixed in advance, not searched. It helped on AJGT and hurt on ASTD, so XLM-R is config-sensitive on small data. Neither configuration beats MARBERT.

### LLM comparison (300-row subset)

Free API quotas capped the LLM test size. All systems are scored on the **same rows**: the full test sets for AJGT and ASTD, and a fixed seeded 300-row stratified sample for LABR and HARD. These 300-row numbers are **not comparable** to the full-test table above (the LABR sample is easier). Invalid replies count as errors, and the rate is shown.

| Dataset (n) | TF-IDF | AraBERT | MARBERT | XLM-R | Qwen | gpt-oss | ALLaM |
|---|---|---|---|---|---|---|---|
| AJGT (180) | 0.872 | 0.911 | 0.928 | 0.869 | 0.878 | 0.830 (10.0% inv) | 0.879 (3.3% inv) |
| ASTD (242) | 0.771 | 0.867 | 0.890 | 0.837 | 0.897 | 0.827 (7.4% inv) | 0.752 (6.6% inv) |
| LABR (300) | 0.883 | 0.873 | 0.917 | 0.870 | 0.880 | 0.840 | 0.845 |
| HARD (300) | 0.950 | 0.953 | 0.960 | 0.953 | 0.927 | 0.866 | 0.927 |
| HARD no-descr. (300) | 0.850 | 0.927 | 0.907 | 0.893 | 0.862 | 0.785 | 0.876 |

CIs on 300 rows are about plus or minus 4 points, so the table is read through paired bootstrap tests on identical rows (`results/paired_llm.csv`). Qwen is not distinguishable from MARBERT on ASTD (0.897 against 0.890 for MARBERT's three-seed mean; paired against seed 0, -0.2 points, interval [-3.5, +3.1]) and is the only LLM clearly above TF-IDF anywhere (ASTD, +12.6). gpt-oss is below MARBERT on all five test sets. ALLaM is below MARBERT on ASTD, LABR and raw HARD, and not distinguishable on AJGT and HARD without the descriptor. On the 300-row LABR and HARD samples no LLM is detectably above TF-IDF, and the LABR sample favours TF-IDF (its score there is 6.8 points above its full-test score, a shift matched by 2 of 10,000 random samples of that shape).

LLM setup: 8 shots (4 positive, 4 negative) from the train split, temperature 0, one-word answer. English instruction chosen over Arabic on 60 validation tweets with a rule fixed in advance (Arabic only if it wins by 5 or more points). Models: `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `allam-2-7b` on Groq, run October 2026.

### Deployed model

One MARBERT fine-tuned on LABR, HARD (descriptor removed, 20,000-review subsample), ASTD and AJGT, single seed. Per-dataset test macro-F1: AJGT 0.933, ASTD 0.869, HARD no-descriptor 0.908, LABR 0.864. Against the single-dataset MARBERT runs it is 1.6 points lower on LABR (paired interval [-2.8, -0.4], so the cost is measurable) and not detectably different on HARD, ASTD (-3.0, -2.3 and -1.1 against the three specialist seeds, seed-0 interval [-6.7, +0.5]) or AJGT.

## Analysis

**HARD leak.** On identical rows and splits, TF-IDF scores 0.941 raw and 0.871 without the leading descriptor. The drop is 5 to 8 points for the LLMs and about 5 for the encoders.

**Dialect analysis.** Dialect labels come from a third-party classifier (`IbrahimAmin/marbertv2-arabic-written-dialect-classifier`), kept at confidence 0.7 or higher. They are predictions, not human annotation, and on HARD they track hotel region as much as dialect. Groups under 50 rows are not reported. Results in `results/dialect_analysis.csv`.

**Error analysis.** An AI assistant (Claude), not a native-speaker annotator, tagged 61 MARBERT errors: all 15 AJGT and all 21 ASTD errors from seed 0, and a random 25 of the 274 LABR errors. Details in `docs/error_analysis.md`. Counts are indicative and need human verification.

## Reproduce

```bash
uv sync
uv run python -m arabic_sentiment.data           # build splits (needs data/raw/hard_extracted, see docs/datasets.md)
uv run python -m arabic_sentiment.baseline
uv run python -m arabic_sentiment.finetune --dataset ajgt --model marbert --seed 0
GROQ_API_KEY=... ./run_llm_queue.sh qwen         # resumable, free-tier daily token caps apply
uv run python -m arabic_sentiment.compare
```

Prediction files and LLM replies contain dataset texts and are not tracked, so `compare.py` and `dialect_analysis.py` need the pipeline rerun first. Experiments are tracked with MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`). Tested on an RTX 5070 laptop GPU (8 GB) with WSL2.

## Serve

```bash
docker build -t arabic-sentiment-api .
docker run --rm -p 8000:8000 -v hf_cache:/home/app/.cache/huggingface arabic-sentiment-api
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "الخدمة كانت سيئة جدا"}'
```

Endpoints: `GET /health`, `POST /predict`, `POST /predict_batch` (max 32 texts), docs at `/docs`. CPU image of about 1.8 GB, weights pulled from the Hub at startup. When the `API_KEY` environment variable is set, `/predict` and `/predict_batch` require an `X-API-Key` header; without it they are open. There is no rate limiting. The public deployment is described under Deployment.

## Deployment

Deployed on Azure Container Apps (consumption plan, scale to zero, 1 vCPU / 2 GiB), image from GitHub Container Registry.
Endpoint: `https://arabic-sentiment-api.niceflower-4ddb1508.uaenorth.azurecontainerapps.io`
`/predict` and `/predict_batch` require an `X-API-Key` header; ask the author for a demo key. `/health` is open.
The first request after idle takes longer while the container starts and loads the model. This is a portfolio demo
with a single shared key and no rate limiting, and it may be taken down when the student credit expires.
Measured cold start from scale-to-zero: about 27 s (container start plus model download). Warm requests: under 1 s round trip.

## Limitations

- Binary task only; neutral and factual text is forced into a polarity.
- The API's `confidence` is the larger softmax probability and it is overconfident: on LABR, rows between 0.9 and 0.99 are right 87.5% of the time and rows between 0.7 and 0.9 are right 65% of the time, while rows at 0.99 or more were right about 98% of the time on all four test sets. Use it as a ranking signal, not a probability.
- ASTD and AJGT test sets are small, and ASTD labels often follow the author's political stance, not wording.
- LLM results are single runs on one prompt, on 300-row subsets for LABR and HARD, and invalid replies count as errors. The public datasets may appear in LLM training data.
- HARD encoder runs use a 20,000-review training subsample, so scores are probably a little below full-data scores.
- LABR reviews are truncated at 256 tokens: about 5% of test reviews for AraBERT and MARBERT. XLM-R's tokenizer produces 27 to 54% more tokens per word, so it truncates more (not measured). A MARBERT run at 128 tokens (13% of reviews truncated) scored the same within noise.
- The HARD descriptor rule also strips 6.0% of rows whose first phrase is a quoted, user-typed title. A stricter rule changed the TF-IDF score by +0.56 points. Encoder and LLM runs were not repeated with it, so their leak gaps may be slightly overstated.
- Dialect labels are predicted, and the dialect analysis cannot separate dialect from domain or hotel region.
- The ASTD copy used differs from a fuller copy: 95 of that copy's 2,483 binary tweets (3.8%) have no exact match in ours, and 31 of ours have none in theirs. Labels agree on every shared tweet, and class balance is the same.
- Not evaluated on Gulf or Maghrebi dialect text or other domains.

## Licenses and citations

Code: MIT. The four datasets do not state licenses, so the model is released for research use and the dataset texts are not redistributed in the current tree (they remain in earlier git history). Cite the original papers (verify details before citing):

- LABR: Aly and Atiya, 2013
- HARD: Elnagar, Khalifa and Einea, 2018, Springer
- ASTD: Nabil, Aly and Atiya, 2015
- AJGT: Alomari, ElSherif and Shaalan, 2017
