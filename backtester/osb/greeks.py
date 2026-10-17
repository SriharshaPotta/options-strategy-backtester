"""Vectorised Black-Scholes-Merton pricing and Greeks."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

ArrayLike = float | np.ndarray

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1_d2(s, k, t, r, sigma, q):
    t = np.maximum(t, 1e-8)
    st = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / st
    return d1, d1 - st


def _is_call(kind) -> np.ndarray:
    return np.asarray(kind) == "call"


def bs_price(s: ArrayLike, k: ArrayLike, t: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0,
             kind: str | np.ndarray = "call") -> np.ndarray:
    s, k, t, sigma = map(np.asarray, (s, k, t, sigma))
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    fs = s * np.exp(-q * t)
    fk = k * np.exp(-r * t)
    call = fs * ndtr(d1) - fk * ndtr(d2)
    put = fk * ndtr(-d2) - fs * ndtr(-d1)
    price = np.where(_is_call(kind), call, put)
    # At expiry fall back to intrinsic value exactly.
    intrinsic = np.where(_is_call(kind), np.maximum(s - k, 0.0), np.maximum(k - s, 0.0))
    return np.where(t <= 0, intrinsic, price)


def bs_greeks(s: ArrayLike, k: ArrayLike, t: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0,
              kind: str | np.ndarray = "call") -> dict[str, np.ndarray]:
    """Delta, gamma, vega (per 1.00 vol) and theta (per year)."""
    s, k, t, sigma = map(np.asarray, (s, k, t, sigma))
    t = np.maximum(t, 1e-8)
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    call = _is_call(kind)
    eq = np.exp(-q * t)
    er = np.exp(-r * t)
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI

    delta = np.where(call, eq * ndtr(d1), eq * (ndtr(d1) - 1.0))
    gamma = eq * pdf / (s * sigma * np.sqrt(t))
    vega = s * eq * pdf * np.sqrt(t)
    common = -s * eq * pdf * sigma / (2.0 * np.sqrt(t))
    theta_call = common - r * k * er * ndtr(d2) + q * s * eq * ndtr(d1)
    theta_put = common + r * k * er * ndtr(-d2) - q * s * eq * ndtr(-d1)
    theta = np.where(call, theta_call, theta_put)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def strike_for_delta(s: float, t: float, r: float, sigma: float, target_delta: float, kind: str,
                     q: float = 0.0) -> float:
    """Strike whose Black-Scholes delta equals `target_delta` (unsigned, e.g. 0.30)."""
    t = max(t, 1e-8)
    a = abs(target_delta) * np.exp(q * t)
    d1 = ndtri(a) if kind == "call" else -ndtri(a)
    return float(s * np.exp(-d1 * sigma * np.sqrt(t) + (r - q + 0.5 * sigma * sigma) * t))
