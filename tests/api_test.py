import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from fastapi.testclient import TestClient
from Applications.api_app import app


# Ignore le warning Pydantic v2
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_minimal():
    payload = {
        "records": [
            {
                "annual_inc": 50000,
                "loan_amnt": 10000,
                "dti": 20.0,
                "int_rate": 12.0,
                "revol_util": 50.0,
                "sub_grade_encoded": 2  # adapte aux features exactes de ton pipeline
            }
        ]
    }
    r = client.post("/predict?threshold=0.4", json=payload)
    assert r.status_code == 200
    out = r.json()[0]
    assert "proba" in out and "label" in out
    assert 0.0 <= out["proba"] <= 1.0
    assert out["label"] in (0, 1)
