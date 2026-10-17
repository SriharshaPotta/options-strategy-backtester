#include "mc/cpu_engine.hpp"

#include <chrono>
#include <cmath>
#include <random>

#include "mc/models.hpp"
#include "mc/payoffs.hpp"
#include "mc/stats.hpp"

namespace mc::cpu {

PriceResult price(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                  std::vector<float>* terminal_out) {
    const auto t0 = std::chrono::steady_clock::now();

    const GbmStep g = make_gbm_step(market, spec.barrier, cfg.n_steps);
    const bool path_dep = is_path_dependent(spec.style);
    const float inv_steps = 1.0f / static_cast<float>(cfg.n_steps);

    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    if (terminal_out) terminal_out->resize(cfg.n_paths);

    double sum = 0.0, sum_sq = 0.0;
    for (uint64_t p = 0; p < cfg.n_paths; ++p) {
        float x = 0.0f;  // log(S_t / S_0)
        float running = 0.0f;
        bool knocked = false;

        for (uint32_t k = 0; k < cfg.n_steps; ++k) {
            x += g.drift + g.diffusion * normal(rng);
            if (path_dep) {
                const float s = g.spot * std::exp(x);
                running += s;
                knocked |= breaches(spec.style, g.barrier, s);
            }
        }

        const float s_t = g.spot * std::exp(x);
        const double payoff = contract_payoff(spec, s_t, running * inv_steps, knocked);
        sum += payoff;
        sum_sq += payoff * payoff;
        if (terminal_out) (*terminal_out)[p] = s_t;
    }

    PriceResult r;
    finalize(r, sum, sum_sq, cfg.n_paths, g.discount);
    r.n_steps = cfg.n_steps;
    r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

}  // namespace mc::cpu
