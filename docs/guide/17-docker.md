# Chapter 17: Docker

## What

A container image that runs the API on a CPU. The image holds the code and the libraries, not the model: weights are downloaded from the Hugging Face Hub when the container starts. Files: `Dockerfile` and `.dockerignore`.

## Why

- **Reproducible runtime:** the same Python, libraries and code on the laptop and on the cloud host.
- **CPU, not GPU.** The target hosts (a free-tier container service) have no GPU. A CPU PyTorch wheel avoids the multi-gigabyte CUDA libraries. `pyproject.toml` pins the CUDA 12.8 index, so the Dockerfile installs its own dependencies and does not use the project lock file (a weakness, below).
- **Weights from the Hub, not baked in.** The image stays small and one model upload serves any number of images. Costs: a download of about 650 MB at every cold start, a dependency on the Hub being reachable, and the image alone is not the whole system.
- **Non-root user and a healthcheck:** standard practice.

## Alternatives

None were tried.

| Choice | Alternative |
|---|---|
| Weights from the Hub at start | Weights copied into the image (a bigger image, no download at start, no Hub dependency) |
| `pip install` with minimum versions | Pinned versions or an export of `uv.lock` (reproducible) |
| `python:3.12-slim` | A distroless or Alpine base (smaller, with more compatibility risk for PyTorch), or a pinned image digest |
| Single stage | Multi-stage build to drop build tools |
| CPU torch | A CUDA image for a GPU host |
| Named volume for the cache | A bind mount, or no cache |
| One uvicorn process | Several workers or gunicorn (more memory per worker) |

## How: the Dockerfile

```dockerfile
FROM python:3.12-slim
```
A Debian-based Python image without compilers or documentation. The tag moves over time, and the image is not pinned to a digest.

```dockerfile
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/app/.cache/huggingface \
    MODEL_DIR=Muhamm-dk/arabic-sentiment-marbert-combined \
    MODEL_VERSION=combined-marbert-s0
```
Unbuffered output so logs appear at once. No pip cache (smaller image). `HF_HOME` is where the Hub client stores downloaded files. `MODEL_DIR` is the Hub repository the app loads (Chapter 16). The revision is not pinned: re-uploading the model changes what a new container loads.

```dockerfile
RUN useradd --create-home app
WORKDIR /app
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
```
A non-root user, and a cache directory it owns. This must come before `USER app`. The ownership matters for volumes (see "What went wrong").

```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install "transformers>=5.0" "fastapi" "uvicorn[standard]" "pydantic>=2"
```
PyTorch from its CPU index, then the serving libraries. Only lower bounds, so a rebuild next month can install newer versions than the ones used to evaluate the model. Whether that changes any prediction was not tested. The check below prints the versions inside the image next to the lock file.

```dockerfile
COPY src/arabic_sentiment/__init__.py src/arabic_sentiment/__init__.py
COPY src/arabic_sentiment/serve src/arabic_sentiment/serve
ENV PYTHONPATH=/app/src
```
Only the package marker and the `serve` folder are copied, not the training code, data or results. `PYTHONPATH` lets `arabic_sentiment.serve.app` import. There is no `__init__.py` inside `serve`; Python treats it as a namespace package and it works.

```dockerfile
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "arabic_sentiment.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Everything after `USER app` runs unprivileged.
- `EXPOSE` documents the port and publishes nothing.
- The healthcheck calls `/health`. `urlopen` raises on an HTTP error, so Python exits non-zero. The 90 s start period allows for the model download.
- `0.0.0.0` so the service is reachable from outside the container. One process, no reload.

`.dockerignore` keeps `.venv`, `data`, `outputs`, `results`, `mlruns`, `mlflow.db`, `notebooks`, `.git` and `__pycache__` out of the build context (8 to 9 kB measured). Only the two `COPY` lines decide what enters the image.

### Build and run

```bash
docker build -t arabic-sentiment-api .
docker run --rm -p 8000:8000 -e API_KEY=testkey \
  -v hf_cache:/home/app/.cache/huggingface arabic-sentiment-api
```

The named volume keeps downloaded weights between runs. `-e API_KEY` switches the key check on; without it the container is open.

## What went wrong

| Item | Detail |
|---|---|
| Wrong image name | I built `arabic_sentiment-api` (underscore) and ran `arabic-sentiment-api` (hyphen). Docker looked for the second on Docker Hub and said "pull access denied". Fix: `docker tag` |
| Cache permission error | The named volume was created empty and owned by root, so the `app` user could not write the model cache. The first fix attempt typed `RUN ...` at the shell prompt, but `RUN` is a Dockerfile instruction. Fix: the `mkdir` and `chown` line before `USER app`, then `docker volume rm hf_cache` (a volume keeps its ownership otherwise) and a rebuild |
| Rebuild needed after a code change | The key-check code reached the image only after rebuilding |
| Docker Spaces unavailable | Hugging Face's Docker and Gradio Spaces could not be selected on the free account, so the deployment moved to Azure (Chapter 18) |

## Numbers

| Item | Value |
|---|---|
| Docker | 29.9.2 on Docker Desktop with WSL integration |
| Image size | 1.8 GB on disk, 387 MB compressed (as Docker Desktop reports them) |
| First build | 94 s: 57.9 s for the PyTorch install and 30.2 s for exporting the image. A rebuild with unchanged layers took 1.6 s |
| Memory | 405 MiB after loading and one request (`docker stats`, against a 21.5 GiB limit on the laptop) |
| Pushed image | `ghcr.io/muhamm-dk/arabic-sentiment-api:0.2.0`, digest `sha256:6f4c049769d6f17aca7bd1a4f73a422936601725bdf9a6fa1b5fbc7415052a66` |
| Local container behavior | `/health` reports `cpu`. A request without a key gets 401, with the key a prediction. The same sentence scored 0.9965 on the container and on the local GPU server |

## Checks (provided, not yet run)

1. **Versions inside the image against the lock file.**

```bash
docker run --rm arabic-sentiment-api python -c "import torch, transformers, fastapi, pydantic, uvicorn; print('torch', torch.__version__, '| transformers', transformers.__version__, '| fastapi', fastapi.__version__, '| pydantic', pydantic.__version__, '| uvicorn', uvicorn.__version__)"
uv pip list 2>/dev/null | grep -i -E "^(torch|transformers|fastapi|pydantic|uvicorn) "
```

   Result: (fill in).

2. **Memory under the largest allowed request** (32 texts of 4,900 characters). `docker stats` is sampled once a second, so the printed maximum is a lower bound.

```bash
python3 -c "import json; print(json.dumps({'texts': ['كتاب جميل ' * 490] * 32}, ensure_ascii=False))" > /tmp/big.json
docker run --rm -d --name api -p 8000:8000 -v hf_cache:/home/app/.cache/huggingface arabic-sentiment-api
sleep 45
rm -f /tmp/mem.log
(while true; do docker stats api --no-stream --format "{{.MemUsage}}" >> /tmp/mem.log; sleep 1; done) & MON=$!
time curl -s -o /dev/null -w "status %{http_code}\n" -X POST localhost:8000/predict_batch -H "Content-Type: application/json" --data-binary @/tmp/big.json
kill $MON
awk '{print $1}' /tmp/mem.log | uniq | tr '\n' ' '
docker stop api
```

   Result: (fill in). The Azure container is set to 2 GiB, which I chose before measuring anything beyond 405 MiB.

## Weaknesses and not done

- **Unpinned dependencies and base image,** so the image is not reproducible. Fix: pin versions (for example from `uv export`) and the base digest.
- **The model revision is not pinned,** so uploading to the Hub repository changes new containers. Fix: a `MODEL_REVISION` variable passed to `from_pretrained(..., revision=...)`.
- **Startup depends on the Hub** being reachable, and downloads unauthenticated (the Hub prints a rate-limit warning).
- **No vulnerability scan,** no tests run inside the build, no read-only filesystem, no resource limits set at the Docker level.
- **Cold start is dominated by the download.** Baking the weights into the image would remove it, at the cost of a roughly 650 MB larger image.
- **The image runs one worker,** so concurrency was not designed or measured.

## Check yourself

1. Why does the Dockerfile not use `pyproject.toml` and `uv.lock`, and what could that cost?
2. Why must the cache directory be created and `chown`ed before `USER app`?
3. What did the volume permission error come from, and why was removing the volume part of the fix?
4. What do the two `COPY` lines leave out, and why does `.dockerignore` matter less than it looks?
5. What are the trade-offs of downloading the weights at start?
6. Why is the memory figure of 405 MiB not a safe sizing number?
7. What happens when someone re-uploads the model to the Hub repository?
