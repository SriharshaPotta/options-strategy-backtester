# Benchmarks

Workload: GBM, S0 = 100, σ = 20%, r = 4%, T = 1y, **252 steps per path**, float path math, double accumulation.

- **GPU time:** simulation kernel + block reduction + result copy, measured with `cudaEvent`. Median of 5 runs after a warm-up. RNG seeding is reported separately.
- **CPU time:** the same discretisation and payoff code on one thread (`std::mt19937_64` + `std::normal_distribution`), Release build.

Reproduce with `notebooks/gpu_benchmark.ipynb`, or locally:

```bash
cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/mc_bench --paths 1048576,2097152,4194304 --steps 252 --json bench.json
```

## GPU vs CPU

> Not yet recorded. Run the notebook on an NVIDIA GPU and paste its markdown table here,
> together with the device name, driver and CPU model it printed.

| case | paths | cpu_ms | gpu_ms | speedup |
|---|---|---|---|---|
| european_call | 1,048,576 | | | |
| european_call | 2,097,152 | | | |
| european_call | 4,194,304 | | | |
| iron_condor | 1,048,576 | | | |
| iron_condor | 2,097,152 | | | |
| iron_condor | 4,194,304 | | | |

## CPU baseline reference

| CPU | case | paths × steps | single-thread time |
|---|---|---|---|
| Intel Core i7-12700H (MSVC 19.4x, /O2) | european_call | 1,048,576 × 252 | 6,752 ms |
| Intel Core i7-12700H (MSVC 19.4x, /O2) | iron_condor | 1,048,576 × 252 | 7,865 ms |
