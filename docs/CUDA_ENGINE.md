# CUDA Monte Carlo engine

Source: [`engine/src/kernels.cu`](../engine/src/kernels.cu) and [`engine/src/gpu_engine.cu`](../engine/src/gpu_engine.cu).

## Launch geometry
`Engine` queries the device once and allocates enough RNG states to fill every SM:

```
threads = multiProcessorCount × (maxThreadsPerMultiProcessor / 256) × 256
```

On a T4 (40 SMs × 1024) that is 40,960 threads. Each call launches `min(threads, round_up(n_paths, 256))` threads in blocks of 256. Every thread walks paths `tid, tid + stride, …` (grid-stride loop), so 1M paths means about 26 paths per thread on a T4.

## Per-thread cuRAND state
```cpp
using RngState = curandStatePhilox4_32_10_t;
curand_init(seed, /*subsequence=*/tid, /*offset=*/0, &states[tid]);
```
- **Philox** is counter-based, so initialisation is O(1) per thread. XORWOW needs a skip-ahead by `tid · 2^67` per thread, which costs far more. Seeding is still timed separately (`rng_init_ms`).
- Each thread owns one subsequence. Streams are independent and the result is reproducible for a given seed and device.
- The state buffer is allocated once and reused. It is re-seeded per call so that bumped revaluations (Greeks) see the same random numbers (common random numbers).
- Inside the kernel the state is copied into registers (`RngState rng = states[tid]`), used for every path the thread owns, and written back once.
- `curand_normal4` returns four normals per call; the step loop is unrolled by four.

## Coalesced memory access
| Access | Pattern | Why it coalesces |
|---|---|---|
| `states[tid]` load/store | once per thread | lanes 0–31 of a warp read 32 adjacent structs, one contiguous span |
| `terminal_out[p]` store | once per path | `p = tid + k·stride`, so on every iteration a warp's lanes hold 32 consecutive `p` |
| contract legs, GBM constants | every step | `__constant__` memory, broadcast to all lanes |
| path state `x`, running sum, barrier flag | every step | registers, never touch DRAM |

By default no per-path data is written at all: payoffs are reduced on the fly. `terminal_out` is only filled when the caller asks for the distribution (dashboard histogram, POP in the backtester).

## Path evolution
```
x ← x + (r − q − σ²/2)Δt + σ√Δt · Z          (float, log space)
S = S0 · __expf(x)                             (only per step for Asian / barrier)
```
European contracts skip the per-step `exp` (template parameter `PathDependent = false`). Path-dependent contracts compute S each step to update the running average and barrier flag.

## Reduction
```
per-thread sum, sum²  (double)
  → __shfl_down_sync tree inside each warp
  → shared memory (8 warps / block)
  → warp 0 reduces the 8 partials
  → atomicAdd(&accum[0..1])  (one pair per block)
```
The host copies two doubles back and computes the discounted mean, standard error and 95% CI (`stats.hpp`).

## Timing
`cudaEventRecord` brackets three phases on the engine's stream: RNG seeding, then simulation + reduction + device-to-host copy of the accumulators. The benchmark warms up once (context creation, module load) and reports the median of 5 runs.

## Validation
`engine/tests/test_gpu_engine.cpp` runs on any machine with a CUDA device and skips otherwise:
- European call within 3 SE of Black-Scholes at 1M paths
- iron condor GPU vs CPU within 3 combined SE
- Asian and up-and-out below vanilla
- reproducibility across calls with the same seed
- mean of terminal prices ≈ S0·e^{(r−q)T}
