# Options Strategy Backtester

**C++ · CUDA · Python · React**

A research tool for listed-options strategies:

- **CUDA Monte Carlo pricing engine** in C++. It simulates 1M+ GBM price paths per run with per-thread cuRAND (Philox) state, coalesced memory access and an on-the-fly warp-shuffle reduction. It supports multi-leg, Asian and barrier payoffs, bump-and-revalue Greeks, and a single-threaded CPU reference for benchmarking.
- **Strategy ranker.** It scores concrete setups (short puts, verticals, iron condors, strangles, straddles…) by **open interest, bid/ask spread and delta fit** on live chains, with risk-neutral POP and max profit/loss.
- **Backtester.** Event-driven entries, profit-target / stop-loss / expiry exits, natural-price fills and commissions. The Monte Carlo engine values every position at entry.
- **React dashboard.** Ranked setups table, backtest P&L / win rate / drawdown / equity curve, and a Monte Carlo pricer with the payoff distribution.

```
dashboard (React) ──HTTP──▶ api (FastAPI) ──▶ osb (Python backtester) ──pybind11──▶ mcengine (C++/CUDA)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/CUDA_ENGINE.md](docs/CUDA_ENGINE.md).

## Performance

The GPU-vs-CPU comparison (1M–4M paths × 252 steps) is produced by [`notebooks/gpu_benchmark.ipynb`](notebooks/gpu_benchmark.ipynb). Results are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quick start

Requirements: CMake ≥ 3.24, a C++17 compiler, Python ≥ 3.10, Node ≥ 20. The CUDA Toolkit ≥ 12 is needed for the GPU backend; without it the engine builds CPU-only.

```bash
# Python env
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Engine (CUDA is picked up automatically when nvcc is on PATH)
cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)"
cmake --build build -j --config Release
ctest --test-dir build -C Release --output-on-failure

# Benchmark
./build/mc_bench --paths 1048576,2097152,4194304 --steps 252

# API + dashboard
uvicorn api.main:app --port 8000
cd dashboard && npm install && npm run dev               # http://localhost:5173
```

The build places `mcengine.*.so` / `.pyd` inside `backtester/osb/`, so `from osb import engine` picks it up.

### Python API

```python
from osb import engine

engine.has_cuda(), engine.device_name()
legs = [("put", 85, 1), ("put", 92, -1), ("call", 108, -1), ("call", 115, 1)]   # iron condor
engine.price(100, 0.04, 0.2, 0.25, legs, n_paths=1 << 20, n_steps=252)
# {'price': ..., 'std_error': ..., 'ci_low': ..., 'ci_high': ..., 'elapsed_ms': ..., 'backend': 'cuda', ...}

from osb import data
from osb.ranker import rank_setups
chain = data.get_live_chain("SPY", data.list_expiries("SPY")[10])
rank_setups(chain, ["iron_condor", "bull_put_spread"]).head()

from osb.backtest import BacktestConfig, run_backtest
res = run_backtest(data.get_history("SPY", "2022-01-01"), BacktestConfig(strategy="iron_condor"))
res.metrics   # win_rate, total_pnl, max_drawdown, sharpe, ...
```

## Ranking

For each template leg the ranker takes the 3 strikes whose |Δ| is closest to the target (e.g. iron condor: 10Δ / 16Δ / 16Δ / 10Δ), builds every valid combination and scores it:

| component | definition | default weight |
|---|---|---|
| open interest | `log1p(min leg OI) / log1p(best in set)` | 0.35 |
| spread | `1 − (Σ half-spreads / |net premium|) / 25%` | 0.35 |
| delta fit | `1 − mean(| |Δ| − target |) / 0.10` | 0.30 |

## Data and limitations

- Live chains (OI, bid/ask, IV) come from yfinance.
- Free historical option chains with OI don't exist, so **backtests run on reconstructed chains**: ATM IV = 30-day realized vol × 1.15, a linear skew + curvature smile, and a spread model that widens out-of-the-money and near expiry. The OI term is dropped from ranking in backtests.
- Results are a model of execution, not a record of it. This is not investment advice.

## Repository layout

```
engine/       C++17 / CUDA engine, GoogleTest suite, benchmark, pybind11 bindings
backtester/   Python package `osb` + pytest suite
api/          FastAPI service
dashboard/    React + Vite + TypeScript + Recharts
notebooks/    GPU benchmark notebook
docs/         plan, roadmap, architecture, CUDA design, benchmarks
```

## License
MIT
