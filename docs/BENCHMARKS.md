# Benchmarks

Workload: GBM, S0 = 100, σ = 20%, r = 4%, T = 1y, **252 steps per path**, float path math, double accumulation.

- **GPU time:** simulation kernel + block reduction + result copy, measured with `cudaEvent`. Median of 5 runs after a warm-up. RNG seeding is reported separately.
- **CPU time:** the same discretisation and payoff code (`std::mt19937_64` + `std::normal_distribution`), Release build, timed with `std::chrono`. Two baselines:
  - `cpu` runs on one thread.
  - `cpu_mt` splits the paths into contiguous blocks across N `std::thread`s, one independent RNG stream each, with N defaulting to every hardware thread.

Reproduce with `notebooks/gpu_benchmark.ipynb`, or locally:

```bash
cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/mc_bench --paths 1048576,2097152,4194304 --steps 252 --json bench.json   # add --threads N to pin cpu_mt
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

## Multi-threaded CPU baseline: Intel Core i7-12700H (20 threads)

Laptop CPU (6 performance + 8 efficiency cores, 20 hardware threads), MSVC `/O2`, 252 steps, one run per configuration.

| case | paths | cpu 1 thread | cpu 20 threads | thread scaling |
|:--|--:|--:|--:|--:|
| european_call | 1,048,576 | 3,075.9 ms | 370.8 ms | 8.3× |
| european_call | 4,194,304 | 11,927.6 ms | 2,443.1 ms | 4.9× |
| iron_condor | 1,048,576 | 4,424.2 ms | 541.3 ms | 8.2× |
| iron_condor | 4,194,304 | 12,963.0 ms | 2,167.4 ms | 6.0× |

Thread scaling (european call, 1M paths):

| threads | 1 | 2 | 4 | 8 | 14 | 20 |
|:--|--:|--:|--:|--:|--:|--:|
| time (ms) | 3,182 | 1,771 | 1,159 | 717 | 541 | 462 |
| speedup | 1.0× | 1.8× | 2.7× | 4.4× | 5.9× | 6.9× |

Scaling flattens beyond the 6 performance cores, and the longer 4M-path runs show thermal throttling. Laptop timings also vary with power state: the same 1-thread, 1M-path run measured 6,752 ms earlier in a different power mode.

## GPU vs multi-threaded CPU

| case | paths | i7-12700H, 20 threads | Tesla T4 | GPU speedup |
|:--|--:|--:|--:|--:|
| european_call | 1,048,576 | 370.8 ms | 6.0 ms | ~62× |
| european_call | 4,194,304 | 2,443.1 ms | 24.0 ms | ~102× |
| iron_condor | 1,048,576 | 541.3 ms | 6.0 ms | ~90× |
| iron_condor | 4,194,304 | 2,167.4 ms | 24.1 ms | ~90× |

This is a cross-machine comparison: the T4 numbers come from Colab and the CPU numbers from the laptop. The Colab host itself exposes only 2 vCPUs. Rerunning the notebook prints a same-host `cpu_mt` column, but with 2 vCPUs that baseline is weak.

## Reading the speedups

- **vs 1 CPU thread: ~1,000×.** Both runs were on the Colab host. This baseline is deliberately simple.
- **vs 20 CPU threads: ~60–100×.** This is the fairer comparison, made across machines.
- **Remaining CPU headroom.** Both CPU baselines spend most of their time in `std::normal_distribution`. A vectorised RNG (e.g. SIMD Philox with Box-Muller) would narrow the gap further.
