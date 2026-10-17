#pragma once

// Host-facing interface to the CUDA engine. Deliberately free of CUDA headers so
// it can be included from plain C++ translation units (bindings, bench, tests).

#include <memory>
#include <string>
#include <vector>

#include "mc/types.hpp"

namespace mc::gpu {

bool available();
std::string device_name();

class Engine {
public:
    explicit Engine(int device = 0);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Prices `spec` with cfg.n_paths paths. RNG states are re-seeded with cfg.seed
    // on every call so that bumped revaluations (Greeks) share random numbers.
    PriceResult price(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                      std::vector<float>* terminal_out = nullptr);

    int threads() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Process-wide engine, lazily created on first use.
Engine& default_engine();

}  // namespace mc::gpu
