"""API tests. Require the trained weights in outputs/combined-marbert-s0 (skipped otherwise)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arabic_sentiment.serve.app import MAX_BATCH, MODEL_DIR, app

pytestmark = pytest.mark.skipif(not Path(MODEL_DIR).exists(), reason="model weights not available")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan, loads model once
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_positive(client):
    r = client.post("/predict", json={"text": "الفندق نظيف والموظفين محترمين جدا"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["truncated"] is False


def test_predict_negative(client):
    r = client.post("/predict", json={"text": "الخدمة كانت سيئة جدا ومش هرجع تاني"})
    assert r.json()["label"] == "negative"


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": "ا" * 5001}])
def test_predict_rejects_invalid(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_long_text_is_truncated(client):
    r = client.post("/predict", json={"text": "كتاب جميل " * 400})
    assert r.status_code == 200 and r.json()["truncated"] is True


def test_batch(client):
    r = client.post("/predict_batch", json={"texts": ["ممتاز جدا", "سيء للغاية"]})
    assert r.status_code == 200
    assert [x["label"] for x in r.json()["results"]] == ["positive", "negative"]


def test_batch_limit_and_blank(client):
    assert client.post("/predict_batch", json={"texts": ["ممتاز"] * (MAX_BATCH + 1)}).status_code == 422
    assert client.post("/predict_batch", json={"texts": ["ممتاز", "  "]}).status_code == 422
    assert client.post("/predict_batch", json={"texts": []}).status_code == 422
