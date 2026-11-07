# Interview notes, part 4: engineering, scenarios, behavioral, weaknesses, live coding

## Serving

**Describe the API.**
FastAPI. `GET /health`, `POST /predict` (one text, 1 to 5,000 characters), `POST /predict_batch` (1 to 32 texts). Each result has `label`, `confidence`, `prob_positive`, `truncated` and `model_version`. 422 on invalid input, 401 on a wrong key.

**How does it load the model?**
Once at startup in a `lifespan` function into a module dictionary, with the model in eval mode on GPU or CPU. Loading per request would cost seconds and 650 MB of reads. The first request after a cold start waits for it.

**How do you validate input?**
Pydantic enforces length bounds. A whitespace-only text passes that, so the endpoint also checks `strip()`. The batch endpoint rechecks every item.

**How does the key check work, and what is wrong with it?**
An `X-API-Key` header is compared in constant time with the `API_KEY` variable. Weaknesses: with the variable unset it fails open, there is one shared key, and no rate limiting. `/health` and `/docs` stay open.

**What do your tests cover?**
13 tests: health, positive and negative sentences (one in Egyptian dialect), invalid payloads, truncation, batches, batch limits, the key check. They do not cover emoji, Latin text, concurrency or the container, and they skip silently if the weights are missing.

**How do you know the API reproduces the model?**
`serve_check.py` ran 1,022 saved test rows through the API's own function on the CPU in fp32. Zero label disagreements, mean probability difference 0.0004. Training evaluated in bf16 on the GPU. It ran in the lock-file environment, not the image. `container_check.py` then repeated it over HTTP on 240 rows against the local container and the Azure app: 0 disagreements.

**What is the performance?**
Memory 405 MiB after loading and one request. Cold start 27.5 s from zero (one sample), warm 0.9 s round trip from the laptop. Throughput and concurrency were not measured.

## Docker

**Walk through the Dockerfile.**
`python:3.12-slim`, a non-root user, a cache directory it owns, CPU PyTorch from its index, then transformers, FastAPI, uvicorn. Only the package marker and `serve` folder are copied. `HEALTHCHECK` calls `/health` with a 90 s start period. The weights are downloaded from the Hub at start.

**What went wrong?**
A named volume was created root-owned, so the non-root user could not write the model cache. The fix was a `mkdir` and `chown` before `USER app`, and removing the old volume. I also built one image name and ran another.

**What is the weakness?**
Versions are unpinned. The image has torch 2.15.0, transformers 5.21.0 and fastapi 0.162.0 against 2.11.0, 5.20.1 and 0.143.3 in my lock file. I then posted 240 saved test rows (60 per dataset) through the HTTP API of the local container and of the Azure app: 0 label disagreements with the saved predictions, mean probability difference 0.00033, maximum 0.0067, identical on both.

**Why weights from the Hub?**
A small image and one upload for any number of images. Cost: a 650 MB download at every cold start, a Hub dependency, and an unpinned revision.

## Deployment

**Why Azure Container Apps?**
It runs the existing image, scales to zero, and the student credit covers it. Hugging Face Docker Spaces were not selectable on the free account. I had claimed they were free, from memory, and was wrong.

**How does scale to zero affect users?**
No traffic means no compute cost, but the first request waits about 27 s while the container starts and downloads the model. A baked-in model or one always-on replica would remove it. The latter would use the free grant in days.

**Security review?**
HTTPS with the platform certificate, one shared key stored as a platform secret, no rate limiting, a public image with no secrets, a non-root user. No vulnerability scan, no alerts, no pinned versions or revisions.

**Cost?**
Student credit of $100 for 12 months, no card. Container Apps has a monthly free grant (180,000 vCPU-seconds and 360,000 GiB-seconds on the pasted page). A budget alert is set. The auto-created Log Analytics workspace may cost something, unchecked.

**What happens when the credit ends?**
The subscription is disabled. I would take the app down or upgrade. Microsoft may remove idle compute after 90 days without activity.

**Secrets handling?**
Tokens and keys go through hidden prompts into environment variables and are unset afterwards. In October a Groq key was typed into a chat. I rotated it, scanned tracked files and results for the key prefix, and cleared the shell history.

## Scenarios (design reasoning, not built)

**Scale to a million requests a day.**
About 12 requests a second on average, with peaks. Batch concurrent requests, run several replicas behind autoscaling with warm minimums, quantize to int8 or use ONNX Runtime, consider a GPU if volume warrants, add a rate limiter and a queue.

**Cut latency?**
Measure first. Options: int8 quantization, ONNX Runtime, truncating to 128 tokens (no detectable loss on LABR, one run), tokenizing once, micro-batching, distilling to a smaller model.

**Monitor in production?**
Request volume, latency, error rate, input length and language mix, label share and confidence distribution over time. Compare with the training distribution. Sample and label a slice regularly. Drift in confidence or label share is the early signal.

**CI/CD?**
Tests plus linting on every push, a build that pins dependencies, a smoke test against the built image, deployment by image digest, and rollback by previous digest. None of it exists here.

**Model versioning?**
Pin the Hub revision, version the image, record the training run (MLflow, JSON results), and keep the model card current. Today the revision is not pinned.

**A/B test a new model?**
Route a fraction of traffic, log both predictions, compare against labeled samples and downstream metrics. Needs labels or a proxy.

**Add a neutral class?**
Three-class labels (ASTD has neutral, LABR's 3 stars), a new head, a macro-F1 over three classes, and re-evaluate. 3-star reviews are ambiguous, so label noise rises.

**Add Gulf dialect?**
Gather Gulf data (ASAD or AraSenTi-Tweet, check access), evaluate before training, add to the combined set, check LABR does not drop.

**Handle long reviews?**
Truncation at 256 tokens covers 95% of LABR. Options: head plus tail, chunk and average, or a long-context model.

**LLM or fine-tuned encoder in production?**
My evidence: MARBERT beat the LLMs on four of five sets. Cost: the encoder serves on a CPU in 405 MiB, with no per-token fees or quota. An LLM needs no labels and adapts with a prompt. For a fixed task with labeled data I would start with the encoder.

**Privacy?**
Request texts are processed and not stored by my code. Platform logs were not reviewed. For real traffic I would define retention, redact PII and review the provider terms.

**"Why not just use an LLM?"**
I did, as a comparison, on three models. Qwen matched MARBERT on ASTD only. The encoder is cheaper, faster and needs no quota. I would use an LLM for labeling or for tasks without labels.

## Behavioral

**What was the hardest problem?**
The HARD leak: scores looked excellent, and I had to prove they were inflated, measure it, and report it without hiding the dataset.

**Mistakes you made?**
Claims from memory that measurement corrected: Docker Spaces being free, XLM-R improving with more training, the quote pattern in HARD, a 490,000-review HARD, "optimum interior on every dataset". The habit I took: write the prediction before running, then score it.

**How did you use AI?**
See part 1. Say what is true, and that you verified outputs.

**How did you prioritize in 3 to 4 weeks?**
Baseline first, one generic training script, small datasets before large ones, subsamples where compute demanded. Dropped: Jais on Kaggle, a hyperparameter search, full-data HARD runs.

**What did you learn?**
Check data before models, report uncertainty, fix decisions before looking at results, and watch long jobs instead of hiding output.

**Why this company?**
Prepare one specific, true sentence for each employer. Check each company's current Arabic AI products first.

## Ten weaknesses and the answer to each

1. **Single seeds on LABR and HARD.** I would run three seeds.
2. **20,000-row HARD subsample.** Rerun on full data, once, on Kaggle.
3. **No hyperparameter search.** A small search per model.
4. **AI-tagged error analysis.** Re-tag the blind sheet myself.
5. **Predicted dialect labels.** Hand-label about 100 rows per dataset to measure the classifier.
6. **No Gulf or Maghrebi data.** Add ASAD or a Saudi set.
7. **300-row LLM samples, one prompt, one run.** More quota or a local model, several prompts.
8. **Unpinned image and revision.** Pin both.
9. **Overconfident scores.** Temperature scaling on validation.
10. **HARD rule over-strips 6% of rows.** Apply the strict rule and rerun the encoders.

## Live coding you should be able to write

```python
def macro_f1(y, p):
    scores = []
    for c in (0, 1):
        tp = ((y == c) & (p == c)).sum()
        fp = ((y != c) & (p == c)).sum()
        fn = ((y == c) & (p != c)).sum()
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return sum(scores) / 2

rng = np.random.default_rng(0)
boots = [macro_f1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
lo, hi = np.percentile(boots, [2.5, 97.5])

n = df.groupby("text").label.transform("nunique")      # drop conflicting texts, keep one copy of the rest
df = df[n == 1].drop_duplicates("text")

def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return scores.softmax(-1) @ v
```

Also be ready to sketch a FastAPI endpoint with a Pydantic model and a dependency, and a training step with autocast, accumulation and clipping (Chapter 9).

## Questions to ask them

- How do you evaluate Arabic NLP systems across dialects, and what data do you use?
- Which dialects matter most for your products, and how do you get labeled data?
- How do you decide between fine-tuned encoders and LLMs for a task?
- How do you monitor models after deployment?
- What does a good first project look like on this team?

End of part 4.
