"""Thin wrapper over the native `mcengine` module.

If the C++/CUDA extension has not been built, a NumPy implementation of the same
GBM scheme is used so the rest of the package (and its tests) still work.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

try:  # built by `cmake --build build` into this package directory
    from . import mcengine as _native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the local build
    _native = None

LegSpec = tuple[str, float, float]  # (call|put|stock, strike, signed qty)


def native_available() -> bool:
    return _native is not None


def has_cuda() -> bool:
    return bool(_native and _native.has_cuda())


def device_name() -> str:
    return _native.device_name() if _native else ""


def _numpy_price(spot, rate, vol, maturity, legs, div, n_paths, n_steps, seed, return_terminal):
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    drift = (rate - div - 0.5 * vol * vol) * dt
    diff = vol * np.sqrt(dt)
    x = np.zeros(n_paths)
    chunk = max(1, 2_000_000 // max(n_paths, 1))  # bound memory per batch of steps
    done = 0
    while done < n_steps:
        m = min(chunk, n_steps - done)
        x += (drift + diff * rng.standard_normal((m, n_paths))).sum(axis=0)
        done += m
    s_t = spot * np.exp(x)
    payoff = payoff_at_expiry(legs, s_t)
    disc = np.exp(-rate * maturity)
    se = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    out = {"price": disc * payoff.mean(), "std_error": se}
    if return_terminal:
        out["terminal"] = s_t.astype(np.float32)
    return out


def price(spot: float, rate: float, vol: float, maturity: float, legs: Sequence[LegSpec], *, div: float = 0.0,
          style: str = "european", barrier: float = 0.0, n_paths: int = 1 << 20, n_steps: int = 252,
          seed: int = 42, backend: str = "auto", return_terminal: bool = False) -> dict[str, Any]:
    legs = [(str(t), float(k), float(q)) for t, k, q in legs]
    if _native is not None:
        return dict(_native.price(spot, rate, vol, maturity, legs, div=div, style=style, barrier=barrier,
                                  n_paths=n_paths, n_steps=n_steps, seed=seed, backend=backend,
                                  return_terminal=return_terminal))
    if style != "european" or backend == "cuda":
        raise RuntimeError("native mcengine module not built; only european/cpu pricing is available")
    t0 = time.perf_counter()
    out = _numpy_price(spot, rate, vol, maturity, legs, div, n_paths, n_steps, seed, return_terminal)
    out.update(ci_low=out["price"] - 1.96 * out["std_error"], ci_high=out["price"] + 1.96 * out["std_error"],
               elapsed_ms=(time.perf_counter() - t0) * 1e3, rng_init_ms=0.0, n_paths=n_paths, n_steps=n_steps,
               backend="numpy")
    return out


def payoff_at_expiry(legs: Sequence[LegSpec], s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    total = np.zeros_like(s)
    for kind, strike, qty in legs:
        if kind == "call":
            total += qty * np.maximum(s - strike, 0.0)
        elif kind == "put":
            total += qty * np.maximum(strike - s, 0.0)
        else:
            total += qty * s
    return total
