#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "mc/models.hpp"
#include "mc/types.hpp"

namespace mc::gpu::detail {

// Philox is counter-based: curand_init is O(1) per thread (no skip-ahead
// matrices as with XORWOW) and curand_normal4 yields four normals per call.
using RngState = curandStatePhilox4_32_10_t;

constexpr int kBlockSize = 256;

void upload_constants(const ContractSpec& spec, const GbmStep& gbm, cudaStream_t stream);
void launch_init_rng(RngState* states, unsigned long long seed, int n_threads, cudaStream_t stream);
void launch_simulate(bool path_dependent, int grid, RngState* states, unsigned long long n_paths, unsigned n_steps,
                     double* accum, float* terminal_out, cudaStream_t stream);

}  // namespace mc::gpu::detail
