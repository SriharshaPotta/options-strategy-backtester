#pragma once

#include <cmath>

#include "mc/types.hpp"

namespace mc {

// Exact GBM discretisation in log space:
//   x_{t+dt} = x_t + (r - q - sigma^2 / 2) dt + sigma sqrt(dt) Z
// Paths evolve in single precision (fast on consumer/datacenter GPUs alike);
// payoffs and accumulators stay in double.
struct GbmStep {
    float drift;
    float diffusion;
    float spot;
    float barrier;
    double discount;
};

inline GbmStep make_gbm_step(const MarketParams& m, double barrier, uint32_t n_steps) {
    const double dt = m.maturity / static_cast<double>(n_steps);
    GbmStep g;
    g.drift = static_cast<float>((m.rate - m.div - 0.5 * m.vol * m.vol) * dt);
    g.diffusion = static_cast<float>(m.vol * std::sqrt(dt));
    g.spot = static_cast<float>(m.spot);
    g.barrier = static_cast<float>(barrier);
    g.discount = std::exp(-m.rate * m.maturity);
    return g;
}

}  // namespace mc
