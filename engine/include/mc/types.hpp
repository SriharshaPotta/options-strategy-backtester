#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define MC_HD __host__ __device__ __forceinline__
#else
#define MC_HD inline
#endif

namespace mc {

enum class OptionType : int32_t { Call = 0, Put = 1, Stock = 2 };

// How the legs are applied to a simulated path.
//   European        - legs settle on S(T)
//   AsianArithmetic - legs settle on the arithmetic average of S over the monitoring dates
//   BarrierUpOut    - legs settle on S(T) unless S ever trades at or above `barrier`
//   BarrierDownOut  - legs settle on S(T) unless S ever trades at or below `barrier`
enum class PayoffStyle : int32_t { European = 0, AsianArithmetic = 1, BarrierUpOut = 2, BarrierDownOut = 3 };

constexpr int kMaxLegs = 8;

struct Leg {
    OptionType type;
    double strike;
    double qty;  // signed: +1 long, -1 short (fractional allowed)
};

struct MarketParams {
    double spot;
    double rate;      // continuously compounded risk-free rate
    double div;       // continuous dividend yield
    double vol;       // annualized volatility
    double maturity;  // years
};

struct ContractSpec {
    PayoffStyle style = PayoffStyle::European;
    double barrier = 0.0;
    int n_legs = 0;
    Leg legs[kMaxLegs] = {};
};

struct SimConfig {
    uint64_t n_paths = 1u << 20;
    uint32_t n_steps = 252;
    uint64_t seed = 42;
};

struct PriceResult {
    double price = 0.0;
    double std_error = 0.0;
    double ci_low = 0.0;
    double ci_high = 0.0;
    double elapsed_ms = 0.0;   // simulation + reduction
    double rng_init_ms = 0.0;  // per-thread RNG seeding (GPU only)
    uint64_t n_paths = 0;
    uint32_t n_steps = 0;
};

}  // namespace mc
