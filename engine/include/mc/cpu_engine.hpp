#pragma once

#include <vector>

#include "mc/types.hpp"

namespace mc::cpu {

// Single-threaded reference engine. Same discretisation and payoff code as the
// CUDA kernels; this is the baseline the GPU speedup is measured against.
PriceResult price(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                  std::vector<float>* terminal_out = nullptr);

}  // namespace mc::cpu
