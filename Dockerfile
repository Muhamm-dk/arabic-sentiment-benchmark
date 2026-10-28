FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/app/.cache/huggingface \
    MODEL_DIR=Muhamm-dk/arabic-sentiment-marbert-combined \
    MODEL_VERSION=combined-marbert-s0

RUN useradd --create-home app
WORKDIR /app
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install "transformers>=5.0" "fastapi" "uvicorn[standard]" "pydantic>=2"

COPY src/arabic_sentiment/__init__.py src/arabic_sentiment/__init__.py
COPY src/arabic_sentiment/serve src/arabic_sentiment/serve
ENV PYTHONPATH=/app/src

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "arabic_sentiment.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
