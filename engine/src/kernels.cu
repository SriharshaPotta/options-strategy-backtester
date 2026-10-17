#include "kernels.cuh"

#include "cuda_utils.cuh"
#include "mc/payoffs.hpp"

namespace mc::gpu::detail {

// Contract and model parameters are identical for every thread and read every
// step, so they live in constant memory (broadcast to the whole warp).
__constant__ ContractSpec c_spec;
__constant__ GbmStep c_gbm;

// One independent Philox subsequence per thread: same seed, subsequence = tid.
__global__ void init_rng_kernel(RngState* __restrict__ states, unsigned long long seed, int n_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_threads) curand_init(seed, static_cast<unsigned long long>(tid), 0ULL, &states[tid]);
}

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

template <bool PathDependent>
__device__ __forceinline__ void advance(float& x, float z, float& running, bool& knocked, const GbmStep& g) {
    x += g.drift + g.diffusion * z;
    if constexpr (PathDependent) {
        const float s = g.spot * __expf(x);
        running += s;
        knocked |= breaches(c_spec.style, g.barrier, s);
    }
}

// Grid-stride Monte Carlo kernel.
//  * RNG state is loaded from global memory once, kept in registers for every
//    path the thread simulates, and written back once at the end. Thread i
//    touches states[i], so a warp reads/writes one contiguous span.
//  * Paths are assigned p = tid, tid + stride, ... so on every iteration the 32
//    lanes of a warp handle 32 consecutive path ids and the optional
//    terminal_out[p] store is a single coalesced transaction.
//  * Payoffs are reduced on the fly (warp shuffle -> shared memory -> one
//    atomicAdd per block); no per-path storage is needed to price.
template <bool PathDependent>
__global__ void __launch_bounds__(kBlockSize)
    simulate_kernel(RngState* __restrict__ states, unsigned long long n_paths, unsigned n_steps,
                    double* __restrict__ accum, float* __restrict__ terminal_out) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    RngState rng = states[tid];
    const GbmStep g = c_gbm;
    const float inv_steps = 1.0f / static_cast<float>(n_steps);

    double sum = 0.0, sum_sq = 0.0;
    for (unsigned long long p = tid; p < n_paths; p += stride) {
        float x = 0.0f, running = 0.0f;
        bool knocked = false;

        unsigned k = 0;
        for (; k + 4 <= n_steps; k += 4) {
            const float4 z = curand_normal4(&rng);
            advance<PathDependent>(x, z.x, running, knocked, g);
            advance<PathDependent>(x, z.y, running, knocked, g);
            advance<PathDependent>(x, z.z, running, knocked, g);
            advance<PathDependent>(x, z.w, running, knocked, g);
        }
        if (k < n_steps) {
            const float4 z = curand_normal4(&rng);
            const float zs[4] = {z.x, z.y, z.z, z.w};
            for (unsigned j = 0; k < n_steps; ++k, ++j) advance<PathDependent>(x, zs[j], running, knocked, g);
        }

        const float s_t = g.spot * __expf(x);
        const double payoff = contract_payoff(c_spec, s_t, running * inv_steps, knocked);
        sum += payoff;
        sum_sq += payoff * payoff;
        if (terminal_out) terminal_out[p] = s_t;
    }

    states[tid] = rng;

    constexpr int kWarps = kBlockSize / 32;
    __shared__ double s_sum[kWarps];
    __shared__ double s_sq[kWarps];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    sum = warp_sum(sum);
    sum_sq = warp_sum(sum_sq);
    if (lane == 0) {
        s_sum[warp] = sum;
        s_sq[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < kWarps ? s_sum[lane] : 0.0;
        sum_sq = lane < kWarps ? s_sq[lane] : 0.0;
        sum = warp_sum(sum);
        sum_sq = warp_sum(sum_sq);
        if (lane == 0) {
            atomicAdd(&accum[0], sum);
            atomicAdd(&accum[1], sum_sq);
        }
    }
}

void upload_constants(const ContractSpec& spec, const GbmStep& gbm, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_spec, &spec, sizeof(ContractSpec), 0, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_gbm, &gbm, sizeof(GbmStep), 0, cudaMemcpyHostToDevice, stream));
}

void launch_init_rng(RngState* states, unsigned long long seed, int n_threads, cudaStream_t stream) {
    const int grid = (n_threads + kBlockSize - 1) / kBlockSize;
    init_rng_kernel<<<grid, kBlockSize, 0, stream>>>(states, seed, n_threads);
    CUDA_CHECK(cudaGetLastError());
}

void launch_simulate(bool path_dependent, int grid, RngState* states, unsigned long long n_paths, unsigned n_steps,
                     double* accum, float* terminal_out, cudaStream_t stream) {
    if (path_dependent)
        simulate_kernel<true><<<grid, kBlockSize, 0, stream>>>(states, n_paths, n_steps, accum, terminal_out);
    else
        simulate_kernel<false><<<grid, kBlockSize, 0, stream>>>(states, n_paths, n_steps, accum, terminal_out);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace mc::gpu::detail
