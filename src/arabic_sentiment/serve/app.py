"""Arabic sentiment API. Run: uv run uvicorn arabic_sentiment.serve.app:app --port 8000"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import secrets
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "outputs/combined-marbert-s0"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "combined-marbert-s0")
MAX_LEN, MAX_BATCH, MAX_CHARS = 256, 32, 5000
LABELS = {0: "negative", 1: "positive"}
state = {}


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(401, "invalid or missing API key")


@asynccontextmanager
async def lifespan(app):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    state["device"] = device
    yield
    state.clear()


app = FastAPI(title="Arabic Sentiment API", version="0.1.0", lifespan=lifespan)


class Item(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class Batch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


@torch.no_grad()
def classify(texts):
    tok, model = state["tok"], state["model"]
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    full = tok(texts, truncation=False)["input_ids"]
    enc = {k: v.to(state["device"]) for k, v in enc.items()}
    probs = model(**enc).logits.float().softmax(-1).cpu()
    out = []
    for p, ids in zip(probs, full):
        i = int(p.argmax())
        out.append({"label": LABELS[i], "confidence": round(float(p[i]), 4),
                    "prob_positive": round(float(p[1]), 4), "truncated": len(ids) > MAX_LEN,
                    "model_version": MODEL_VERSION})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "device": state.get("device"), "model_version": MODEL_VERSION}


@app.post("/predict", dependencies=[Depends(require_key)])
def predict(item: Item):
    if not item.text.strip():
        raise HTTPException(422, "text is blank")
    return classify([item.text])[0]


@app.post("/predict_batch", dependencies=[Depends(require_key)])
def predict_batch(batch: Batch):
    if any(not t.strip() or len(t) > MAX_CHARS for t in batch.texts):
        raise HTTPException(422, f"each text must be non-blank and at most {MAX_CHARS} characters")
    return {"results": classify(batch.texts)}
