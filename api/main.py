"""HTTP API for the dashboard.

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from osb import data, engine
from osb.backtest import BacktestConfig, run_backtest
from osb.chains import expiry_for_dte, synthetic_chain
from osb.ranker import RankWeights, rank_setups
from osb.strategies import STRATEGIES

app = FastAPI(title="Options Strategy Backtester", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def _clean(obj):
    """Replace NaN/inf with None so responses are valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


@app.get("/api/engine")
def engine_info():
    return {"native": engine.native_available(), "cuda": engine.has_cuda(), "device": engine.device_name()}


@app.get("/api/strategies")
def strategies():
    return [
        {"key": s.key, "label": s.label, "dte": s.dte, "description": s.description, "tags": list(s.tags),
         "legs": [{"kind": leg.kind, "side": "long" if leg.side > 0 else "short", "target_delta": leg.target_delta}
                  for leg in s.legs]}
        for s in STRATEGIES.values()
    ]


@app.get("/api/expiries")
def expiries(ticker: str = "SPY"):
    try:
        return data.list_expiries(ticker)
    except Exception as exc:  # network / unknown ticker
        raise HTTPException(502, f"could not load expiries for {ticker}: {exc}") from exc


@app.get("/api/rank")
def rank(ticker: str = "SPY", expiry: str | None = None, strategy: list[str] | None = Query(None),
         source: str = Query("live", pattern="^(live|synthetic)$"), top_n: int = Query(25, ge=1, le=200),
         w_oi: float = 0.35, w_spread: float = 0.35, w_delta: float = 0.30, iv: float = 0.18, spot: float = 100.0):
    weights = RankWeights(open_interest=w_oi, spread=w_spread, delta=w_delta)
    if strategy:
        unknown = sorted(set(strategy) - set(STRATEGIES))
        if unknown:
            raise HTTPException(400, f"unknown strategies: {unknown}")
    try:
        if source == "live":
            exp = expiry or _nearest_expiry(ticker, 30)
            chain = data.get_live_chain(ticker, exp)
        else:
            exp_date = date.fromisoformat(expiry) if expiry else expiry_for_dte(date.today(), 30)
            chain = synthetic_chain(ticker.upper(), spot, date.today(), exp_date, iv)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"could not build chain for {ticker}: {exc}") from exc

    ranked = rank_setups(chain, strategy, weights, top_n=top_n)
    return _clean({"ticker": chain.underlying, "spot": chain.spot, "expiry": chain.expiry.isoformat(),
                   "dte": chain.dte, "source": chain.source, "setups": ranked.to_dict("records")})


def _nearest_expiry(ticker: str, dte: int) -> str:
    exps = data.list_expiries(ticker)
    if not exps:
        raise HTTPException(404, f"no listed options for {ticker}")
    today = date.today()
    return min(exps, key=lambda e: abs((date.fromisoformat(e) - today).days - dte))


class BacktestRequest(BaseModel):
    ticker: str = "SPY"
    strategy: str = "iron_condor"
    start: str = "2022-01-01"
    end: str | None = None
    dte: int | None = Field(None, ge=1, le=365)
    entry_every: int = Field(5, ge=1, le=60)
    max_open: int = Field(4, ge=1, le=20)
    profit_target: float = Field(0.5, gt=0)
    stop_loss: float = Field(2.0, gt=0)
    contracts: int = Field(1, ge=1, le=100)
    mc_paths: int = Field(200_000, ge=1_000, le=4_000_000)


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy '{req.strategy}'")
    try:
        close = data.get_history(req.ticker, req.start, req.end)
    except Exception as exc:
        raise HTTPException(502, f"could not load history for {req.ticker}: {exc}") from exc
    cfg = BacktestConfig(**req.model_dump())
    try:
        result = run_backtest(close, cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _clean(result.to_dict())


class Leg(BaseModel):
    kind: str = Field(pattern="^(call|put|stock)$")
    strike: float = Field(ge=0)
    qty: float


class PriceRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    rate: float = 0.04
    vol: float = Field(0.2, gt=0)
    maturity: float = Field(0.25, gt=0)
    div: float = 0.0
    legs: list[Leg] = Field(default_factory=lambda: [Leg(kind="call", strike=100.0, qty=1.0)])
    style: str = Field("european", pattern="^(european|asian|up_and_out|down_and_out)$")
    barrier: float = 0.0
    n_paths: int = Field(1 << 20, ge=1_000, le=16_000_000)
    n_steps: int = Field(252, ge=1, le=2_000)
    seed: int = 42
    backend: str = Field("auto", pattern="^(auto|cuda|cpu)$")
    bins: int = Field(60, ge=10, le=200)


@app.post("/api/price")
def price(req: PriceRequest):
    legs = [(leg.kind, leg.strike, leg.qty) for leg in req.legs]
    try:
        r = engine.price(req.spot, req.rate, req.vol, req.maturity, legs, div=req.div, style=req.style,
                         barrier=req.barrier, n_paths=req.n_paths, n_steps=req.n_steps, seed=req.seed,
                         backend=req.backend, return_terminal=True)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    terminal = r.pop("terminal")
    counts, edges = np.histogram(terminal, bins=req.bins)
    pnl = engine.payoff_at_expiry(legs, terminal)
    r["histogram"] = [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(c)} for i, c in enumerate(counts)]
    r["payoff_curve"] = [{"spot": float(s), "payoff": float(p)}
                         for s, p in zip(edges, engine.payoff_at_expiry(legs, edges))]
    r["prob_positive_payoff"] = float((pnl > 0).mean())
    return _clean(r)
