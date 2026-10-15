"""API tests via FastAPI TestClient (no network port needed)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import churn_model

VALID = {
    "CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 42,
    "Tenure": 3, "Balance": 75000.0, "NumOfProducts": 2, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 100000.0,
}


@pytest.fixture(scope="module")
def client():
    # The context manager runs lifespan, so the model is loaded at startup.
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert 0.5 <= body["model_roc_auc"] <= 1.0


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert isinstance(body["churn"], bool)


def test_predict_validation_error(client):
    r = client.post("/predict", json={**VALID, "Age": 5})  # < 18 -> 422
    assert r.status_code == 422


def test_predict_unknown_geography(client):
    r = client.post("/predict", json={**VALID, "Geography": "Mars"})  # not in the enum -> 422
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"customers": [VALID, VALID]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_503_without_model(monkeypatch):
    # No context manager: lifespan does not run, and the loaded model is hidden.
    monkeypatch.setattr(churn_model, "_pipeline", None)
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "degraded"
    r = c.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]
    assert c.post("/predict/batch", json={"customers": [VALID]}).status_code == 503


def test_predict_batch_empty(client):
    r = client.post("/predict/batch", json={"customers": []})
    assert r.status_code == 422


def test_predict_extra_field(client):
    r = client.post("/predict", json={**VALID, "Surname": "Smith"})
    assert r.status_code == 422


def test_predict_missing_field(client):
    body = {k: v for k, v in VALID.items() if k != "Balance"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422
