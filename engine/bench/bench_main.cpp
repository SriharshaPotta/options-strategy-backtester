// CPU (single-threaded) vs CUDA timing for the Monte Carlo engine.
//
//   mc_bench [--paths 1048576,2097152,4194304] [--steps 252] [--reps 5] [--cpu-reps 3] [--json out.json]
//
// GPU time = simulation kernel + reduction + result copy (cudaEvent). RNG seeding
// is reported separately. CPU time = std::chrono around the whole loop.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mc/cpu_engine.hpp"
#include "mc/gpu_engine.cuh"
#include "mc/stats.hpp"

namespace {

struct Case {
    std::string name;
    mc::ContractSpec spec;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::vector<uint64_t> parse_list(const std::string& s) {
    std::vector<uint64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoull(item));
    return out;
}

mc::ContractSpec european_call(double k) {
    mc::ContractSpec s;
    s.n_legs = 1;
    s.legs[0] = {mc::OptionType::Call, k, 1.0};
    return s;
}

mc::ContractSpec iron_condor(double s0) {
    mc::ContractSpec s;
    s.n_legs = 4;
    s.legs[0] = {mc::OptionType::Put, 0.85 * s0, 1.0};
    s.legs[1] = {mc::OptionType::Put, 0.92 * s0, -1.0};
    s.legs[2] = {mc::OptionType::Call, 1.08 * s0, -1.0};
    s.legs[3] = {mc::OptionType::Call, 1.15 * s0, 1.0};
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<uint64_t> sizes{1u << 20, 1u << 21, 1u << 22};
    uint32_t steps = 252;
    int reps = 5, cpu_reps = 3;
    std::string json_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--paths") sizes = parse_list(next());
        else if (a == "--steps") steps = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--cpu-reps") cpu_reps = std::stoi(next());
        else if (a == "--json") json_path = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    const mc::MarketParams m{100.0, 0.04, 0.0, 0.2, 1.0};
    const std::vector<Case> cases{{"european_call", european_call(100.0)}, {"iron_condor", iron_condor(100.0)}};

    const bool has_gpu = mc::gpu::available();
    const std::string device = has_gpu ? mc::gpu::device_name() : "none";
    std::printf("device: %s | steps: %u | gpu reps: %d | cpu reps: %d\n", device.c_str(), steps, reps, cpu_reps);
    std::printf("%-14s %10s %12s %12s %10s %10s %9s %s\n", "case", "paths", "cpu_ms", "gpu_ms", "rng_ms", "speedup",
                "bs_ref", "agree");

    std::ostringstream js;
    js << "{\n  \"device\": \"" << device << "\",\n  \"steps\": " << steps << ",\n  \"results\": [";
    bool first = true;

    for (const auto& c : cases) {
        const double ref = mc::black_scholes(c.spec, m);
        for (uint64_t n : sizes) {
            const mc::SimConfig cfg{n, steps, 1234};

            std::vector<double> cpu_ms;
            mc::PriceResult cpu_r;
            for (int r = 0; r < cpu_reps; ++r) {
                cpu_r = mc::cpu::price(m, c.spec, cfg);
                cpu_ms.push_back(cpu_r.elapsed_ms);
            }

            double gpu_med = NAN, rng_med = NAN;
            bool agree = true;
            mc::PriceResult gpu_r;
            if (has_gpu) {
                auto& eng = mc::gpu::default_engine();
                eng.price(m, c.spec, cfg);  // warm-up: context, module load, constant upload
                std::vector<double> gpu_ms, rng_ms;
                for (int r = 0; r < reps; ++r) {
                    gpu_r = eng.price(m, c.spec, cfg);
                    gpu_ms.push_back(gpu_r.elapsed_ms);
                    rng_ms.push_back(gpu_r.rng_init_ms);
                }
                gpu_med = median(gpu_ms);
                rng_med = median(rng_ms);
                const double tol = 3.0 * std::hypot(cpu_r.std_error, gpu_r.std_error);
                agree = std::abs(cpu_r.price - gpu_r.price) <= tol;
            }

            const double cpu_med = median(cpu_ms);
            const double speedup = cpu_med / gpu_med;
            std::printf("%-14s %10llu %12.1f %12.2f %10.2f %9.1fx %9.4f %s\n", c.name.c_str(),
                        static_cast<unsigned long long>(n), cpu_med, gpu_med, rng_med, speedup, ref,
                        has_gpu ? (agree ? "yes" : "NO") : "-");

            js << (first ? "\n" : ",\n") << "    {\"case\": \"" << c.name << "\", \"paths\": " << n
               << ", \"cpu_ms\": " << cpu_med << ", \"cpu_price\": " << cpu_r.price
               << ", \"cpu_se\": " << cpu_r.std_error << ", \"bs_reference\": " << ref;
            if (has_gpu)
                js << ", \"gpu_ms\": " << gpu_med << ", \"gpu_rng_init_ms\": " << rng_med
                   << ", \"gpu_price\": " << gpu_r.price << ", \"gpu_se\": " << gpu_r.std_error
                   << ", \"speedup\": " << speedup << ", \"prices_agree\": " << (agree ? "true" : "false");
            js << "}";
            first = false;
            if (!agree) std::fprintf(stderr, "warning: CPU/GPU prices disagree beyond 3 SE for %s\n", c.name.c_str());
        }
    }
    js << "\n  ]\n}\n";

    if (!json_path.empty()) {
        std::ofstream(json_path) << js.str();
        std::printf("wrote %s\n", json_path.c_str());
    }
    return 0;
}
