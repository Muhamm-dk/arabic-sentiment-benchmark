"""API key tests. Require the model weights (skipped otherwise)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arabic_sentiment.serve.app import MODEL_DIR, app

pytestmark = pytest.mark.skipif(not Path(MODEL_DIR).exists(), reason="model weights not available")
BODY = {"text": "ممتاز جدا"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_open_when_no_key_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert client.post("/predict", json=BODY).status_code == 200


def test_rejects_missing_and_wrong_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/predict", json=BODY).status_code == 401
    assert client.post("/predict", json=BODY, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict_batch", json={"texts": ["ممتاز"]}).status_code == 401


def test_accepts_correct_key_and_health_stays_open(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/predict", json=BODY, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200
