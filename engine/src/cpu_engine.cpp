#include "mc/cpu_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

#include "mc/models.hpp"
#include "mc/payoffs.hpp"
#include "mc/stats.hpp"

namespace mc::cpu {

namespace {

struct Moments {
    double sum = 0.0;
    double sum_sq = 0.0;
};

// Simulates paths [begin, end) with one RNG stream. Shared by the single- and
// multi-threaded entry points so both run exactly the same per-path code.
Moments simulate_range(const GbmStep& g, const ContractSpec& spec, uint32_t n_steps, uint64_t begin, uint64_t end,
                       uint64_t seed, float* terminal_out) {
    const bool path_dep = is_path_dependent(spec.style);
    const float inv_steps = 1.0f / static_cast<float>(n_steps);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    Moments m;
    for (uint64_t p = begin; p < end; ++p) {
        float x = 0.0f;  // log(S_t / S_0)
        float running = 0.0f;
        bool knocked = false;

        for (uint32_t k = 0; k < n_steps; ++k) {
            x += g.drift + g.diffusion * normal(rng);
            if (path_dep) {
                const float s = g.spot * std::exp(x);
                running += s;
                knocked |= breaches(spec.style, g.barrier, s);
            }
        }

        const float s_t = g.spot * std::exp(x);
        const double payoff = contract_payoff(spec, s_t, running * inv_steps, knocked);
        m.sum += payoff;
        m.sum_sq += payoff * payoff;
        if (terminal_out) terminal_out[p] = s_t;
    }
    return m;
}

// Independent stream per worker; worker 0 keeps the caller's seed so a
// one-thread run reproduces the single-threaded engine exactly.
uint64_t stream_seed(uint64_t seed, unsigned worker) { return seed + 0x9E3779B97F4A7C15ull * worker; }

}  // namespace

unsigned default_threads() { return std::max(1u, std::thread::hardware_concurrency()); }

PriceResult price(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                  std::vector<float>* terminal_out) {
    const auto t0 = std::chrono::steady_clock::now();
    const GbmStep g = make_gbm_step(market, spec.barrier, cfg.n_steps);
    if (terminal_out) terminal_out->resize(cfg.n_paths);

    const Moments m = simulate_range(g, spec, cfg.n_steps, 0, cfg.n_paths, stream_seed(cfg.seed, 0),
                                     terminal_out ? terminal_out->data() : nullptr);

    PriceResult r;
    finalize(r, m.sum, m.sum_sq, cfg.n_paths, g.discount);
    r.n_steps = cfg.n_steps;
    r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

PriceResult price_mt(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg, unsigned n_threads,
                     std::vector<float>* terminal_out) {
    const auto t0 = std::chrono::steady_clock::now();
    const GbmStep g = make_gbm_step(market, spec.barrier, cfg.n_steps);
    if (terminal_out) terminal_out->resize(cfg.n_paths);
    float* out = terminal_out ? terminal_out->data() : nullptr;

    const unsigned workers = static_cast<unsigned>(
        std::max<uint64_t>(1, std::min<uint64_t>(n_threads ? n_threads : default_threads(), cfg.n_paths)));

    // Contiguous block per worker: no shared writes, terminal_out slices are disjoint.
    std::vector<Moments> partial(workers);
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) {
        const uint64_t begin = cfg.n_paths * w / workers;
        const uint64_t end = cfg.n_paths * (w + 1) / workers;
        pool.emplace_back([&, w, begin, end] {
            partial[w] = simulate_range(g, spec, cfg.n_steps, begin, end, stream_seed(cfg.seed, w), out);
        });
    }
    for (auto& t : pool) t.join();

    Moments total;
    for (const auto& m : partial) {
        total.sum += m.sum;
        total.sum_sq += m.sum_sq;
    }

    PriceResult r;
    finalize(r, total.sum, total.sum_sq, cfg.n_paths, g.discount);
    r.n_steps = cfg.n_steps;
    r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

}  // namespace mc::cpu
