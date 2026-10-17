#pragma once

#include <vector>

#include "mc/types.hpp"

namespace mc::cpu {

// Single-threaded reference engine. Same discretisation and payoff code as the
// CUDA kernels; this is the baseline the GPU speedup is measured against.
PriceResult price(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                  std::vector<float>* terminal_out = nullptr);

// Multi-threaded engine: paths split into contiguous blocks, one std::thread and
// one independent mt19937_64 stream per block. n_threads = 0 uses every hardware
// thread. With n_threads = 1 the result is bit-identical to price().
PriceResult price_mt(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                     unsigned n_threads = 0, std::vector<float>* terminal_out = nullptr);

unsigned default_threads();

}  // namespace mc::cpu
