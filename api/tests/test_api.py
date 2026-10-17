import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import main
from osb import data

client = TestClient(main.app)


def test_engine_and_strategies():
    assert set(client.get("/api/engine").json()) == {"native", "cuda", "device"}
    keys = {s["key"] for s in client.get("/api/strategies").json()}
    assert "iron_condor" in keys


def test_rank_synthetic():
    r = client.get("/api/rank", params={"ticker": "TEST", "source": "synthetic", "spot": 250, "iv": 0.2,
                                        "strategy": ["iron_condor", "short_put"], "top_n": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["setups"]) == 5
    assert {s["strategy"] for s in body["setups"]} <= {"iron_condor", "short_put"}
    assert body["setups"][0]["min_oi"] is None  # synthetic chains carry no OI


def test_rank_rejects_unknown_strategy():
    r = client.get("/api/rank", params={"source": "synthetic", "strategy": ["nope"]})
    assert r.status_code == 400


def test_price_returns_histogram():
    r = client.post("/api/price", json={"n_paths": 20000, "n_steps": 4, "backend": "cpu", "bins": 20})
    assert r.status_code == 200
    body = r.json()
    assert len(body["histogram"]) == 20
    assert sum(b["count"] for b in body["histogram"]) == 20000
    assert body["price"] > 0


def test_backtest_uses_history(monkeypatch):
    idx = pd.bdate_range("2023-01-02", periods=220)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.008, len(idx)))), index=idx)
    monkeypatch.setattr(data, "get_history", lambda *a, **k: close)
    r = client.post("/api/backtest", json={"ticker": "FAKE", "strategy": "bull_put_spread", "mc_paths": 5000})
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["n_trades"] > 0
    assert len(body["equity"]) > 100
