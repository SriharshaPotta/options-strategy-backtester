import numpy as np
import pandas as pd

from osb.backtest import BacktestConfig, run_backtest


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)), dtype=float)


def test_flat_market_short_premium_always_wins():
    close = _series(np.full(260, 100.0))
    cfg = BacktestConfig(ticker="FLAT", strategy="bull_put_spread", mc_paths=20_000, mc_backend="cpu")
    res = run_backtest(close, cfg)
    assert res.metrics["n_trades"] > 10
    # Positions still open on the last bar are force-closed at the natural price
    # (paying the spread), so only check trades that ran their course.
    finished = res.trades[res.trades["reason"] != "end"]
    assert (finished["pnl"] > 0).all()
    assert res.metrics["win_rate"] > 0.9
    assert res.metrics["total_pnl"] > 0
    assert (res.trades["mc_pop"] > 0.5).all()


def test_crash_hits_stop_loss():
    up = np.full(80, 100.0)
    crash = np.linspace(100.0, 60.0, 40)
    close = _series(np.concatenate([up, crash, np.full(40, 60.0)]))
    res = run_backtest(close, BacktestConfig(ticker="CRASH", strategy="short_put", mc_paths=20_000, mc_backend="cpu"))
    assert "stop" in set(res.trades["reason"])
    assert res.metrics["total_pnl"] < 0
    assert res.metrics["max_drawdown"] < 0


def test_result_serialises():
    close = _series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200))))
    res = run_backtest(close, BacktestConfig(strategy="iron_condor", mc_paths=10_000, mc_backend="cpu"))
    d = res.to_dict()
    assert d["equity"] and d["trades"]
    assert set(d["metrics"]) >= {"win_rate", "total_pnl", "sharpe", "max_drawdown"}
