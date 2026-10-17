# Benchmarks

Workload: GBM, S0 = 100, σ = 20%, r = 4%, T = 1y, **252 steps per path**, float path math, double accumulation.

- **GPU time:** simulation kernel + block reduction + result copy, measured with `cudaEvent`. Median of 5 runs after a warm-up. RNG seeding is reported separately.
- **CPU time:** the same discretisation and payoff code on one thread (`std::mt19937_64` + `std::normal_distribution`), Release build.

Reproduce with `notebooks/gpu_benchmark.ipynb`, or locally:

```bash
cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/mc_bench --paths 1048576,2097152,4194304 --steps 252 --json bench.json
```

## GPU vs CPU — NVIDIA Tesla T4 (Google Colab)

Measured with `notebooks/gpu_benchmark.ipynb` on a Colab T4 runtime; CPU column is the same Colab host, one thread.

| case | paths | cpu_ms | gpu_ms | speedup |
|:--|--:|--:|--:|--:|
| european_call | 1,048,576 | 6,245.9 | 6.0 | 1,035× |
| european_call | 2,097,152 | 12,325.7 | 12.0 | 1,023× |
| european_call | 4,194,304 | 25,006.0 | 24.0 | 1,043× |
| iron_condor | 1,048,576 | 6,145.9 | 6.0 | 1,018× |
| iron_condor | 2,097,152 | 12,303.4 | 12.1 | 1,020× |
| iron_condor | 4,194,304 | 25,136.8 | 24.1 | 1,044× |

Throughput on the T4 is about **44 billion path-steps per second** (4.19M paths × 252 steps in 24 ms), and time scales linearly with path count.

**Accuracy check** (iron condor, 1M paths × 252 steps, through the Python bindings):

| backend | price | std error | sim time |
|:--|--:|--:|--:|
| CUDA (Tesla T4) | −3.8280 | 0.0030 | 3.3 ms |
| CPU (1 thread) | −3.8238 | 0.0030 | 6,296.4 ms |
| Black-Scholes | −3.8268 | — | — |

**Reading the speedup.** The baseline is deliberately simple: one thread with `std::mt19937_64` and `std::normal_distribution`, and the normal draw dominates its cost. A multi-threaded, SIMD-vectorised CPU implementation would narrow the gap considerably. The ~1,000× figure is a GPU-vs-single-thread comparison, not a GPU-vs-best-CPU comparison.

## Other CPU baselines

| CPU | case | paths × steps | single-thread time |
|---|---|---|---|
| Intel Core i7-12700H (MSVC 19.4x, /O2) | european_call | 1,048,576 × 252 | 6,752 ms |
| Intel Core i7-12700H (MSVC 19.4x, /O2) | iron_condor | 1,048,576 × 252 | 7,865 ms |
