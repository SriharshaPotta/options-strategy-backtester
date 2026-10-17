#pragma once

#include "mc/types.hpp"

namespace mc {

struct Greeks {
    double price;
    double delta;
    double gamma;
    double vega;  // per 1.00 change in vol (divide by 100 for per vol point)
};

// Bump-and-revalue with common random numbers: every revaluation reuses cfg.seed,
// so the noise largely cancels in the finite differences.
template <typename Pricer>
Greeks bump_greeks(Pricer&& pricer, const MarketParams& m, const ContractSpec& spec, const SimConfig& cfg,
                   double rel_spot_bump = 0.01, double vol_bump = 0.01) {
    const double h = m.spot * rel_spot_bump;
    MarketParams up = m, dn = m, vu = m, vd = m;
    up.spot += h;
    dn.spot -= h;
    vu.vol += vol_bump;
    vd.vol -= vol_bump;

    const double p0 = pricer(m, spec, cfg).price;
    const double pu = pricer(up, spec, cfg).price;
    const double pd = pricer(dn, spec, cfg).price;
    const double pvu = pricer(vu, spec, cfg).price;
    const double pvd = pricer(vd, spec, cfg).price;

    return Greeks{p0, (pu - pd) / (2.0 * h), (pu - 2.0 * p0 + pd) / (h * h), (pvu - pvd) / (2.0 * vol_bump)};
}

}  // namespace mc
