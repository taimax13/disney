import os
os.environ["USE_SAMPLE_DATA"] = "1" # ensure sample data
from fastapi.testclient import TestClient
from ..app import app


def test_health():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"




def test_ask_basic():
    client = TestClient(app)
    r = client.get("/ask", params={"q": "What do visitors from Australia say about Disneyland in Hong Kong?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["stats"]["used"] >= 1