"""Market data via yfinance with a local parquet cache."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .chains import Chain, from_yfinance

CACHE_DIR = Path(os.environ.get("OSB_CACHE", Path.home() / ".osb_cache"))


def _yf():
    import yfinance as yf  # imported lazily: slow import, network-only use

    return yf


def get_history(ticker: str, start: str | date, end: str | date | None = None, refresh: bool = False) -> pd.Series:
    """Daily adjusted closes indexed by date."""
    end = end or date.today().isoformat()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker.upper()}_{start}_{end}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["close"]

    df = _yf().download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no price history for {ticker} between {start} and {end}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns a column MultiIndex for single tickers too
        close = close.iloc[:, 0]
    close = close.rename("close").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.to_frame().to_parquet(path)
    return close


def list_expiries(ticker: str) -> list[str]:
    return list(_yf().Ticker(ticker).options)


def get_spot(ticker: str) -> float:
    t = _yf().Ticker(ticker)
    price = t.fast_info.get("lastPrice") if hasattr(t, "fast_info") else None
    if not price:
        price = t.history(period="5d")["Close"].iloc[-1]
    return float(price)


def get_live_chain(ticker: str, expiry: str, rate: float = 0.04) -> Chain:
    yf = _yf()
    tk = yf.Ticker(ticker)
    oc = tk.option_chain(expiry)
    spot = get_spot(ticker)
    return from_yfinance(ticker.upper(), spot, date.today(), datetime.strptime(expiry, "%Y-%m-%d").date(),
                         oc.calls, oc.puts, rate=rate)
