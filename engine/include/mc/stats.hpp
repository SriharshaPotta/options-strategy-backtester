#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "mc/types.hpp"

namespace mc {

// Turns raw payoff moments into a discounted price with standard error and 95% CI.
inline void finalize(PriceResult& r, double sum, double sum_sq, uint64_t n, double discount) {
    const double nd = static_cast<double>(n);
    const double mean = sum / nd;
    const double var = std::max(0.0, (sum_sq / nd - mean * mean) * nd / std::max(1.0, nd - 1.0));
    r.price = discount * mean;
    r.std_error = discount * std::sqrt(var / nd);
    r.ci_low = r.price - 1.96 * r.std_error;
    r.ci_high = r.price + 1.96 * r.std_error;
    r.n_paths = n;
}

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Closed-form Black-Scholes-Merton, used as the reference in tests and benchmarks.
inline double black_scholes(OptionType type, const MarketParams& m, double strike) {
    if (type == OptionType::Stock) return m.spot * std::exp(-m.div * m.maturity);
    const double st = m.vol * std::sqrt(m.maturity);
    const double d1 = (std::log(m.spot / strike) + (m.rate - m.div + 0.5 * m.vol * m.vol) * m.maturity) / st;
    const double d2 = d1 - st;
    const double df_s = m.spot * std::exp(-m.div * m.maturity);
    const double df_k = strike * std::exp(-m.rate * m.maturity);
    if (type == OptionType::Call) return df_s * norm_cdf(d1) - df_k * norm_cdf(d2);
    return df_k * norm_cdf(-d2) - df_s * norm_cdf(-d1);
}

inline double black_scholes(const ContractSpec& spec, const MarketParams& m) {
    double total = 0.0;
    for (int i = 0; i < spec.n_legs; ++i) total += spec.legs[i].qty * black_scholes(spec.legs[i].type, m, spec.legs[i].strike);
    return total;
}

}  // namespace mc
