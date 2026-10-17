"""Option chains: live (yfinance) and reconstructed (Black-Scholes on realized vol).

Free historical option chains with open interest do not exist, so backtests run
on chains rebuilt from the underlying's price history:

  * ATM implied vol  = rolling realized vol x a variance-risk-premium multiplier
  * smile            = linear put skew + curvature in log-moneyness / sqrt(T)
  * bid/ask          = modelled half-spread widening out-of-the-money and near expiry
  * open interest    = NaN (unknown) -> the ranker drops the OI term for these chains
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .greeks import bs_greeks, bs_price

CHAIN_COLUMNS = ["kind", "strike", "bid", "ask", "mid", "iv", "delta", "open_interest", "volume"]


@dataclass
class Chain:
    underlying: str
    spot: float
    asof: date
    expiry: date
    rate: float
    frame: pd.DataFrame  # CHAIN_COLUMNS
    source: str = "synthetic"

    @property
    def t(self) -> float:
        return max((self.expiry - self.asof).days, 0) / 365.0

    @property
    def dte(self) -> int:
        return (self.expiry - self.asof).days

    def side(self, kind: str) -> pd.DataFrame:
        return self.frame[self.frame["kind"] == kind]


def strike_step(spot: float) -> float:
    if spot < 25:
        return 0.5
    if spot < 200:
        return 1.0
    if spot < 1000:
        return 5.0
    return 10.0


def expiry_for_dte(asof: date, dte: int) -> date:
    """First Friday on or after asof + dte."""
    target = asof + timedelta(days=dte)
    return target + timedelta(days=(4 - target.weekday()) % 7)


@dataclass(frozen=True)
class SmileModel:
    skew: float = -0.10     # dIV / d(log-moneyness / sqrt T)
    curvature: float = 0.05
    floor: float = 0.05

    def iv(self, atm_iv: float, spot: float, strikes: np.ndarray, t: float) -> np.ndarray:
        m = np.log(strikes / spot) / np.sqrt(max(t, 1 / 365))
        return np.maximum(atm_iv * (1.0 + self.skew * m + self.curvature * m * m), self.floor)


@dataclass(frozen=True)
class SpreadModel:
    base_pct: float = 0.03      # half-spread as fraction of mid near ATM
    otm_pct: float = 0.15       # extra fraction for far out-of-the-money strikes
    short_dte_pct: float = 0.04
    min_half: float = 0.01      # dollars
    tick: float = 0.01

    def half_spread(self, mid: np.ndarray, delta: np.ndarray, dte: int) -> np.ndarray:
        wing = np.clip(0.5 - np.abs(delta), 0.0, 0.5) * 2.0  # 0 ATM -> 1 far OTM
        pct = self.base_pct + self.otm_pct * wing ** 2 + self.short_dte_pct * (7.0 / max(dte, 7))
        return np.maximum(mid * pct, self.min_half)


def synthetic_chain(underlying: str, spot: float, asof: date, expiry: date, atm_iv: float, rate: float = 0.04,
                    div: float = 0.0, n_strikes_each_side: int = 40, smile: SmileModel = SmileModel(),
                    spreads: SpreadModel = SpreadModel()) -> Chain:
    step = strike_step(spot)
    atm = round(spot / step) * step
    strikes = atm + step * np.arange(-n_strikes_each_side, n_strikes_each_side + 1)
    strikes = strikes[strikes > 0]
    t = max((expiry - asof).days, 0) / 365.0
    iv = smile.iv(atm_iv, spot, strikes, t)

    frames = []
    for kind in ("put", "call"):
        mid = bs_price(spot, strikes, t, rate, iv, div, kind)
        delta = bs_greeks(spot, strikes, t, rate, iv, div, kind)["delta"]
        half = spreads.half_spread(mid, delta, (expiry - asof).days)
        bid = np.maximum(np.floor((mid - half) / spreads.tick) * spreads.tick, 0.0)
        ask = np.ceil((mid + half) / spreads.tick) * spreads.tick
        frames.append(pd.DataFrame({
            "kind": kind, "strike": strikes, "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "iv": iv,
            "delta": delta, "open_interest": np.nan, "volume": np.nan,
        }))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["ask"] >= 0.05].reset_index(drop=True)  # drop untradeable far wings
    return Chain(underlying, float(spot), asof, expiry, rate, frame, source="synthetic")


def from_yfinance(underlying: str, spot: float, asof: date, expiry: date, calls: pd.DataFrame, puts: pd.DataFrame,
                  rate: float = 0.04, div: float = 0.0) -> Chain:
    """Normalise a yfinance `option_chain()` result and attach Black-Scholes deltas."""
    t = max((expiry - asof).days, 1) / 365.0
    frames = []
    for kind, raw in (("call", calls), ("put", puts)):
        if raw is None or raw.empty:
            continue
        df = pd.DataFrame({
            "kind": kind,
            "strike": raw["strike"].astype(float),
            "bid": raw["bid"].fillna(0.0).astype(float),
            "ask": raw["ask"].fillna(0.0).astype(float),
            "iv": raw["impliedVolatility"].astype(float),
            "open_interest": raw["openInterest"].fillna(0).astype(float),
            "volume": raw["volume"].fillna(0).astype(float),
        })
        df = df[(df["ask"] > 0) & (df["iv"] > 0.01)]
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df["delta"] = bs_greeks(spot, df["strike"].to_numpy(), t, rate, df["iv"].to_numpy(), div, kind)["delta"]
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)[CHAIN_COLUMNS] if frames else pd.DataFrame(columns=CHAIN_COLUMNS)
    return Chain(underlying, float(spot), asof, expiry, rate, frame, source="yfinance")


def realized_vol(close: pd.Series, window: int = 30) -> pd.Series:
    """Annualised close-to-close realized volatility."""
    return np.log(close).diff().rolling(window).std() * np.sqrt(252.0)
