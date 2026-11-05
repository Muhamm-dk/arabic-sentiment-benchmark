# Chapter 16: Serving

## What

`src/arabic_sentiment/serve/app.py` wraps the combined MARBERT (Chapter 15) in a FastAPI service with two prediction endpoints, a health check and an optional key check, plus 13 tests (`tests/test_api.py`, `tests/test_auth.py`).

## Why

- **It is the step from a notebook result to something another program can call,** and the part employers for engineering roles look for.
- **FastAPI** validates input from type annotations (Pydantic), generates interactive documentation at `/docs`, and runs under uvicorn. I did not compare it with other frameworks.
- **Load the model once at startup,** not per request. Loading takes seconds and 650 MB of reads.
- **Validate input tightly** so a malformed or enormous request fails fast and cheaply.

## Alternatives

None were tried.

| Choice | Alternative |
|---|---|
| FastAPI | Flask, Django REST, BentoML, TorchServe, a managed inference endpoint (which skips the service code) |
| PyTorch fp32 on CPU | ONNX Runtime, int8 dynamic quantization, bf16. Often faster on CPU, with an unknown accuracy cost |
| One forward pass per request | Server-side micro-batching across concurrent requests |
| Synchronous endpoints | Async endpoints with a worker queue |
| One shared secret in a header | Per-client keys, OAuth, a gateway with rate limiting |
| Reject texts over 5,000 characters | Truncate on the server |
| Return the raw confidence | A calibrated probability, or an "uncertain" flag |

## The contract

| Endpoint | Request | Response | Key needed |
|---|---|---|---|
| `GET /health` | none | `{"status": "ok", "device": "cpu" or "cuda", "model_version": ...}` | no |
| `POST /predict` | `{"text": "..."}`, 1 to 5,000 characters, not blank | one result object | yes, if `API_KEY` is set |
| `POST /predict_batch` | `{"texts": [...]}`, 1 to 32 items, each non-blank and at most 5,000 characters | `{"results": [result, ...]}` | yes, if `API_KEY` is set |

Result object: `label` ("positive" or "negative"), `confidence` (the larger of the two probabilities, 4 decimals), `prob_positive`, `truncated` (true if the text exceeded 256 tokens) and `model_version`.

Errors: 401 for a missing or wrong key, 422 for any validation failure (the FastAPI default `{"detail": ...}`). `/docs` and `/openapi.json` are open.

## How: the code

### Configuration

```python
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "outputs/combined-marbert-s0"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "combined-marbert-s0")
MAX_LEN, MAX_BATCH, MAX_CHARS = 256, 32, 5000
```

Everything deployment-specific comes from environment variables (the Dockerfile sets `MODEL_DIR` to the Hub repository). The default points at the local weights relative to the working directory, so `uvicorn` and `pytest` must run from the repository root. A Hub id passes through `Path` unchanged on Linux, and `from_pretrained` accepts it.

### Key check

```python
def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(401, "invalid or missing API key")
```

- Attached to the two prediction routes with `dependencies=[Depends(require_key)]`. `/health` stays open for the container healthcheck.
- It reads the environment variable on every request, so the tests can switch the key on and off with `monkeypatch`.
- `compare_digest` compares in constant time, so response timing does not leak how many characters match.
- **It fails open:** if `API_KEY` is unset, anyone can call the API. That is convenient locally and dangerous if a deployment forgets the variable. A fail-closed design would refuse to start. Not implemented.
- One shared secret for all callers, no rate limiting, no per-client revocation.

### Startup

```python
@asynccontextmanager
async def lifespan(app):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    state["device"] = device
    yield
    state.clear()
```

The model and tokenizer load once, before the first request is accepted, into a module-level dictionary. `.eval()` turns dropout off. No requests are served while the model loads, so on a cold start the first request waits (27 s measured in Chapter 18). Weights are fp32 and no warm-up request runs.

### Prediction

```python
@torch.no_grad()
def classify(texts):
    tok, model = state["tok"], state["model"]
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    full = tok(texts, truncation=False)["input_ids"]
    enc = {k: v.to(state["device"]) for k, v in enc.items()}
    probs = model(**enc).logits.float().softmax(-1).cpu()
    ...
        out.append({"label": LABELS[i], "confidence": round(float(p[i]), 4),
                    "prob_positive": round(float(p[1]), 4), "truncated": len(ids) > MAX_LEN, ...})
```

- Same maximum length as training (256, special tokens included), padded to the longest text in the request, as in `finetune.py`.
- `full` tokenizes every text a second time without truncation just to decide the `truncated` flag. That doubles tokenization work and makes the tokenizer print its long-sequence warning. A cheaper way exists (check the overflow while tokenizing once). I did not measure the cost.
- The model sees the same input as in training except for precision: training evaluated under bf16 autocast, serving runs fp32. The training-serving check below measures the effect.
- The endpoints are plain `def` functions, so FastAPI runs them in its thread pool (memory). Concurrent requests share one model and compete for the CPU. Not load-tested.

### Two validation layers

Pydantic checks the length bounds (`Field(min_length=1, max_length=MAX_CHARS)`, list length 1 to 32). A whitespace-only text passes that check, so `predict` also checks `item.text.strip()` and returns 422 with "text is blank". The batch endpoint checks each item again, because a list of strings has no per-item constraint.

## Tests

13 tests, run from the repository root, 4.5 s with the model loaded (the first version with 10 tests took 7 s):

| File | Tests |
|---|---|
| `test_api.py` | health; one positive and one negative sentence (the second in Egyptian dialect); four invalid payloads (empty, blank, missing field, 5,001 characters); a long text sets `truncated`; a two-text batch returns the right labels; batch limits (33 items, a blank item, an empty list) |
| `test_auth.py` | open when no key is configured; 401 for a missing or wrong key on both routes; the right key is accepted and `/health` stays open |

Weaknesses, from reading them:
- **The weights are required, or everything skips.** Without `outputs/combined-marbert-s0` pytest reports all 13 as skipped. A CI run without weights would look green. With a Hub id in `MODEL_DIR` they skip as well (a Hub id is not an existing path).
- **They run on the GPU when one exists,** while the deployed container runs on the CPU. The CPU path was checked by hand (the same sentence returned 0.9965 on both) and by the training-serving check.
- **Three hand-written sentences** carry the model assertions. They break if a retrained model changes its mind.
- **Nothing covers:** emoji, Latin script, text of digits only, diacritics, concurrency, timeouts, or the container itself.
- `ruff` was installed as a dev tool and was never run on the repository. `import secrets` sits after `import torch` in `app.py`, which an import-order check would flag.

## Training-serving check

`serve_check.py` runs the API's `classify` on 300 test rows per dataset on the CPU and compares them with the saved predictions. Results (CPU, fp32, the API's own `classify` function through the FastAPI test client in batches of 32, local weights in `outputs/combined-marbert-s0`):

| Dataset | Rows | Macro-F1, saved predictions (bf16, GPU) | Macro-F1, API code (fp32, CPU) | Rows over 256 tokens |
|---|---|---|---|---|
| AJGT | 180 (all) | 0.9333 | 0.9333 | 0 |
| ASTD | 242 (all) | 0.8691 | 0.8691 | 1 |
| HARD, descriptor removed | 300 (random sample, seed 0) | 0.9231 | 0.9231 | 0 |
| LABR | 300 (random sample, seed 0) | 0.8532 | 0.8532 | 17 |

- Label disagreements: 0 of 1,022 rows. Probability difference between the saved bf16 values and the fp32 API values: mean 0.00041, maximum 0.0102.
- The HARD and LABR rows are random 300-row samples drawn inside the script. They are not the LLM samples, so their scores differ from the full-test scores (0.9078 and 0.8641).
- Test rows over 5,000 characters, which the API answers with 422: 4 of 13,412, all in LABR (0.18% of its test set).
- LABR truncation: 17 of 300 (5.7%), against 4.6% over the full test set (Chapter 8). The difference is inside sampling noise (about 1.2 points at n=300).
- Predictions scored: at most 0.5% disagreements (held, 0), mean difference under 0.005 and maximum under 0.05 (held), 5 to 25 rows over 5,000 characters all in LABR (half held, all in LABR but only 4).
- Not covered: the HTTP layer and the key check, the Hub copy of the weights, the Docker image and the Azure deployment (each was checked on one sentence only, 0.9845 or 0.9965), CPU speed, and any quantized or ONNX variant.


Predictions made before running: at most 0.5% label disagreements, a mean probability difference under 0.005 and a maximum under 0.05, and between 5 and 25 test rows over 5,000 characters, all in LABR.

## What went wrong

| Item | Detail |
|---|---|
| Key check not applied at first | The patch was in a message, but the file on disk had no `require_key`, and pytest showed 10 tests, not 13. A grep returned 0. The patch was then applied, and the container test proved it: 401 without a key, a normal answer with one |
| Two kinds of 422 message | An empty text gets Pydantic's "String should have at least 1 character", a whitespace-only text gets "text is blank". I had expected one message |
| Fails open | See above. A forgotten variable means an open API |
| `confidence` is overconfident | See Chapter 15. The field is a ranking signal |
| Texts over 5,000 characters are rejected, not truncated | Only the first 256 tokens matter, so a long review could be accepted and cut. The limit exists to bound the tokenization work. See the check above for how many test rows it would reject |

## Not done

- Rate limiting, request logging, metrics, structured errors, CORS settings (not needed without a browser caller).
- Load testing and a measurement of memory under the largest request (Chapter 17 gives a check).
- An "uncertain" flag, or calibrated probabilities.
- A fail-closed key check, key rotation, a per-client key.
- Pinning the model revision (Chapter 17).
- Tests against the built container and against the deployed URL.

## Check yourself

1. Why is the model loaded in `lifespan` and not inside the endpoint?
2. What does `require_key` do when `API_KEY` is unset, and why is that a risk?
3. Why does `classify` tokenize twice, and what would you change?
4. What is different between the training predictions and the API predictions, and how does `serve_check.py` test it?
5. Why can all 13 tests skip and still look like a pass?
6. Why is the `confidence` field not a probability of being correct?
7. Why does a batch request need its own per-item check?
8. What happens to a 6,000-character review, and what design alternative exists?
