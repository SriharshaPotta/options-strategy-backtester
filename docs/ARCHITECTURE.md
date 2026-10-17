# Architecture

```
┌───────────────────────────┐      HTTP/JSON       ┌──────────────────────────────┐
│ dashboard/ (React + Vite) │ ───────────────────▶ │ api/main.py (FastAPI)        │
│  Ranked Setups            │   /api/rank          │                              │
│  Backtest                 │   /api/backtest      │                              │
│  MC Pricer                │   /api/price         │                              │
└───────────────────────────┘                      └──────────────┬───────────────┘
                                                                  │ Python
                                ┌─────────────────────────────────▼───────────────────────────────┐
                                │ backtester/osb                                                  │
                                │  data.py ── yfinance history / live chains (parquet cache)      │
                                │  chains.py ─ normalise live chains, reconstruct historical ones │
                                │  greeks.py ─ vectorised Black-Scholes, strike-for-delta         │
                                │  strategies.py ─ templates as delta targets                     │
                                │  ranker.py ─ OI / spread / delta scoring, POP, max P/L          │
                                │  backtest.py ─ entries, marks, exits, fills, fees               │
                                │  metrics.py ─ win rate, P&L, drawdown, Sharpe                   │
                                │  engine.py ─ wrapper over the native module (NumPy fallback)    │
                                └─────────────────────────────────┬───────────────────────────────┘
                                                                  │ pybind11 (GIL released)
                                ┌─────────────────────────────────▼───────────────────────────────┐
                                │ engine/ (C++17 / CUDA)  → mcengine.*.pyd / .so                  │
                                │  gpu_engine.cu + kernels.cu ── CUDA backend                     │
                                │  cpu_engine.cpp ───────────── single-thread reference           │
                                │  payoffs.hpp / models.hpp ─── shared __host__ __device__ code   │
                                └─────────────────────────────────────────────────────────────────┘
```

## Data flow

**Ranking (live).**
1. `data.get_live_chain` pulls a yfinance chain: strikes, bid/ask, IV and open interest.
2. `chains.from_yfinance` attaches Black-Scholes deltas.
3. For each strategy template, `ranker` takes the 3 strikes nearest each leg's delta target, builds every valid combination and scores it.

**Backtest.**
1. `data.get_history` loads adjusted closes.
2. Every *N* trading days, `chains.synthetic_chain` rebuilds a chain from spot and 30-day realized vol × premium, with modelled skew and spreads.
3. The ranker picks the best setup (OI weight dropped, because reconstructed chains have no OI), and the position opens at the natural price.
4. The CUDA engine prices the legs' expiry payoff and returns terminal prices. From these the backtester computes model value, edge and probability of profit.
5. Positions are marked daily on the same chain model and close at the profit target, stop loss or expiry settlement.

**Pricer.** `/api/price` runs the engine with `return_terminal=True` and returns price, SE, CI and timing, plus a histogram of S(T) and the payoff curve.

## Shared code between CPU and GPU
`payoffs.hpp`, `models.hpp` and `types.hpp` are compiled by both the host compiler and nvcc (`MC_HD` expands to `__host__ __device__`). The two backends therefore use identical payoff logic and discretisation, and the benchmark compares like with like.

## Build matrix
| Environment | Engine backends | Python module |
|---|---|---|
| NVIDIA GPU + CUDA toolkit | CUDA + CPU | `mcengine` (backend `auto` → CUDA) |
| No CUDA compiler (CI, laptops) | CPU (`gpu_stub.cpp`) | `mcengine` (backend `auto` → CPU) |
| Module not built | — | NumPy fallback in `osb.engine` (European only) |
