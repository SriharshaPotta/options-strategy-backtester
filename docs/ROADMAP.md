# Roadmap

## M1 — Scaffold
- [x] Repo layout, CMake project, Python package, Vite app
- [x] CI: CPU engine + tests, CUDA compile check, Python tests + lint, dashboard lint + build
- [x] Design docs (PLAN, ARCHITECTURE, CUDA_ENGINE)

## M2 — CPU reference engine
- [x] GBM log-space stepping, multi-leg payoffs, Asian and barrier styles
- [x] Price, standard error, 95% CI
- [x] Bump-and-revalue Greeks with common random numbers
- [x] GoogleTest suite against Black-Scholes

## M3 — CUDA engine
- [x] Per-thread Philox cuRAND state, seeded by a dedicated kernel
- [x] Grid-stride simulation kernel with coalesced state/terminal access
- [x] Warp-shuffle + shared-memory block reduction
- [x] RAII device buffers, `CUDA_CHECK`, stream + event timing
- [x] pybind11 module with `auto | cuda | cpu` backends
- [x] `mc_bench` CPU vs GPU benchmark (JSON output)

## M4 — Backtester
- [x] yfinance data + parquet cache
- [x] Live chain normalisation and reconstructed historical chains
- [x] Strategy templates by delta target
- [x] Ranker: open interest, spread, delta fit; POP and max profit/loss
- [x] Event-driven backtest with profit target, stop loss, expiry settlement, commissions
- [x] Metrics: P&L, win rate, profit factor, drawdown, Sharpe, CAGR

## M5 — API + dashboard
- [x] FastAPI endpoints: engine, strategies, expiries, rank, backtest, price
- [x] Ranked setups table (sortable, leg drill-down)
- [x] Backtest view: KPIs, equity curve, per-trade P&L, trade log
- [x] MC pricer: leg editor, payoff distribution histogram

## M6 — GPU benchmark
- [ ] Run `notebooks/gpu_benchmark.ipynb` on an NVIDIA GPU and record results in BENCHMARKS.md

## Later
- [ ] Heston stochastic-volatility paths
- [ ] Antithetic variates / control variates
- [ ] Multi-GPU split across devices
- [ ] Historical chains from a paid data provider (real OI in backtests)
