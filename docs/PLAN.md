# Project Plan

## Goal
An options strategy research tool with three parts:

1. **CUDA Monte Carlo pricing engine (C++17/CUDA)** that simulates 1M+ GBM price paths per run. It uses per-thread cuRAND state and coalesced memory access, and is benchmarked against a single-threaded CPU implementation of the same scheme.
2. **Python backtester** that builds option chains, ranks concrete strategy setups by **open interest, bid/ask spread and delta fit**, and backtests strategies with realistic fills, exits and P&L accounting.
3. **React dashboard** for ranking setups, running backtests (P&L, win rate, drawdown, equity curve) and exploring Monte Carlo payoff distributions.

## Design decisions

| Area | Decision | Why |
|---|---|---|
| Path model | GBM, exact log-space step, configurable steps (default 252) | Closed-form reference for validation; steps make Asian/barrier payoffs meaningful |
| Precision | `float` path evolution, `double` payoff accumulation | Single-precision throughput on GPUs; accumulators keep the mean/SE stable |
| RNG | `curandStatePhilox4_32_10`, one subsequence per thread, `curand_normal4` | O(1) init per thread, four normals per call, reproducible by seed |
| Memory | State array indexed by thread id; grid-stride path assignment; optional terminal prices stored at `out[path]` | Each warp touches a contiguous span, so loads and stores coalesce |
| Reduction | Warp shuffle → shared memory → one `atomicAdd` per block | No per-path storage needed to price |
| Contracts | Up to 8 legs (call/put/stock) in `__constant__` memory; European, arithmetic Asian, up/down-and-out | Covers spreads, condors, straddles and exotics with one kernel |
| Greeks | Bump-and-revalue with common random numbers | Reuses the pricer; noise cancels in differences |
| Bindings | pybind11 module `mcengine`, GIL released during simulation | Backtester and API call the engine directly |
| Fallback | CMake builds a CPU-only engine when no CUDA compiler is present; Python falls back to NumPy if the module is not built | CI runners and contributors without a GPU can still build and test |
| Market data | yfinance live chains (OI, bid/ask, IV) for ranking; reconstructed chains (realized vol × premium, skew, spread model) for history | Free historical chains with OI are not available; the limitation is stated in the UI and docs |
| API / UI | FastAPI + React/Vite/TypeScript + Recharts | Small, typed, fast to iterate |

## Milestones
See [ROADMAP.md](ROADMAP.md).

## Verification
- Engine: MC prices within 3 standard errors of Black-Scholes for calls, puts, verticals and condors; put-call parity; Asian and knock-out prices below vanilla; bump delta within 0.01 of analytic; GPU/CPU agreement.
- Python: Black-Scholes values and Greeks, strike-for-delta round trip, ranker ordering on crafted chains, deterministic backtests (flat market → short premium wins; crash → stop loss).
- API: endpoint tests with FastAPI `TestClient`.
- Benchmarks: `notebooks/gpu_benchmark.ipynb`. Only measured numbers go into [BENCHMARKS.md](BENCHMARKS.md).
