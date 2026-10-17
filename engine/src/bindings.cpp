#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mc/cpu_engine.hpp"
#include "mc/gpu_engine.cuh"
#include "mc/greeks.hpp"

namespace py = pybind11;

namespace {

using LegTuple = std::tuple<std::string, double, double>;  // (type, strike, qty)

mc::OptionType parse_type(const std::string& s) {
    if (s == "call") return mc::OptionType::Call;
    if (s == "put") return mc::OptionType::Put;
    if (s == "stock") return mc::OptionType::Stock;
    throw std::invalid_argument("unknown leg type '" + s + "' (expected call|put|stock)");
}

mc::PayoffStyle parse_style(const std::string& s) {
    if (s == "european") return mc::PayoffStyle::European;
    if (s == "asian") return mc::PayoffStyle::AsianArithmetic;
    if (s == "up_and_out") return mc::PayoffStyle::BarrierUpOut;
    if (s == "down_and_out") return mc::PayoffStyle::BarrierDownOut;
    throw std::invalid_argument("unknown style '" + s + "' (expected european|asian|up_and_out|down_and_out)");
}

mc::ContractSpec make_spec(const std::vector<LegTuple>& legs, const std::string& style, double barrier) {
    if (legs.empty()) throw std::invalid_argument("at least one leg is required");
    if (legs.size() > static_cast<size_t>(mc::kMaxLegs))
        throw std::invalid_argument("at most " + std::to_string(mc::kMaxLegs) + " legs are supported");
    mc::ContractSpec spec;
    spec.style = parse_style(style);
    spec.barrier = barrier;
    spec.n_legs = static_cast<int>(legs.size());
    for (size_t i = 0; i < legs.size(); ++i)
        spec.legs[i] = mc::Leg{parse_type(std::get<0>(legs[i])), std::get<1>(legs[i]), std::get<2>(legs[i])};
    return spec;
}

enum class Backend { Cuda, Cpu, CpuMt };

Backend resolve(const std::string& backend) {
    if (backend == "cuda") {
        if (!mc::gpu::available()) throw std::runtime_error("backend='cuda' requested but no CUDA device is available");
        return Backend::Cuda;
    }
    if (backend == "cpu") return Backend::Cpu;
    if (backend == "cpu_mt") return Backend::CpuMt;
    if (backend == "auto") return mc::gpu::available() ? Backend::Cuda : Backend::CpuMt;
    throw std::invalid_argument("unknown backend '" + backend + "' (expected auto|cuda|cpu|cpu_mt)");
}

const char* name(Backend b) {
    switch (b) {
        case Backend::Cuda: return "cuda";
        case Backend::Cpu: return "cpu";
        default: return "cpu_mt";
    }
}

unsigned thread_count(Backend b, unsigned threads) {
    if (b == Backend::Cpu) return 1;
    if (b == Backend::CpuMt) return threads ? threads : mc::cpu::default_threads();
    return 0;
}

mc::PriceResult run(Backend b, unsigned threads, const mc::MarketParams& m, const mc::ContractSpec& spec,
                    const mc::SimConfig& cfg, std::vector<float>* terminal = nullptr) {
    switch (b) {
        case Backend::Cuda: return mc::gpu::default_engine().price(m, spec, cfg, terminal);
        case Backend::Cpu: return mc::cpu::price(m, spec, cfg, terminal);
        default: return mc::cpu::price_mt(m, spec, cfg, threads, terminal);
    }
}

py::dict price(double spot, double rate, double vol, double maturity, const std::vector<LegTuple>& legs, double div,
               const std::string& style, double barrier, uint64_t n_paths, uint32_t n_steps, uint64_t seed,
               const std::string& backend, bool return_terminal, unsigned threads) {
    const mc::MarketParams m{spot, rate, div, vol, maturity};
    const mc::ContractSpec spec = make_spec(legs, style, barrier);
    const mc::SimConfig cfg{n_paths, n_steps, seed};
    const Backend b = resolve(backend);

    std::vector<float> terminal;
    mc::PriceResult r;
    {
        py::gil_scoped_release release;
        r = run(b, threads, m, spec, cfg, return_terminal ? &terminal : nullptr);
    }

    py::dict out;
    out["price"] = r.price;
    out["std_error"] = r.std_error;
    out["ci_low"] = r.ci_low;
    out["ci_high"] = r.ci_high;
    out["elapsed_ms"] = r.elapsed_ms;
    out["rng_init_ms"] = r.rng_init_ms;
    out["n_paths"] = r.n_paths;
    out["n_steps"] = r.n_steps;
    out["backend"] = name(b);
    out["threads"] = thread_count(b, threads);
    if (return_terminal) out["terminal"] = py::array_t<float>(terminal.size(), terminal.data());
    return out;
}

py::dict greeks(double spot, double rate, double vol, double maturity, const std::vector<LegTuple>& legs, double div,
                const std::string& style, double barrier, uint64_t n_paths, uint32_t n_steps, uint64_t seed,
                const std::string& backend, unsigned threads) {
    const mc::MarketParams m{spot, rate, div, vol, maturity};
    const mc::ContractSpec spec = make_spec(legs, style, barrier);
    const mc::SimConfig cfg{n_paths, n_steps, seed};
    const Backend b = resolve(backend);

    mc::Greeks g;
    {
        py::gil_scoped_release release;
        g = mc::bump_greeks([&](const mc::MarketParams& mm, const mc::ContractSpec& s, const mc::SimConfig& c) {
            return run(b, threads, mm, s, c);
        }, m, spec, cfg);
    }

    py::dict out;
    out["price"] = g.price;
    out["delta"] = g.delta;
    out["gamma"] = g.gamma;
    out["vega"] = g.vega;
    out["backend"] = name(b);
    return out;
}

}  // namespace

PYBIND11_MODULE(mcengine, mod) {
    mod.doc() = "Monte Carlo options pricing engine (CUDA with single-threaded CPU reference)";
    mod.attr("__version__") = "0.1.0";

    mod.def("has_cuda", &mc::gpu::available, "True if a CUDA device is usable");
    mod.def("device_name", &mc::gpu::device_name, "Name of the active CUDA device, or '' if none");
    mod.def("cpu_threads", &mc::cpu::default_threads, "Hardware threads used by backend='cpu_mt' by default");

    mod.def("price", &price, py::arg("spot"), py::arg("rate"), py::arg("vol"), py::arg("maturity"), py::arg("legs"),
            py::arg("div") = 0.0, py::arg("style") = "european", py::arg("barrier") = 0.0,
            py::arg("n_paths") = 1u << 20, py::arg("n_steps") = 252, py::arg("seed") = 42,
            py::arg("backend") = "auto", py::arg("return_terminal") = false, py::arg("threads") = 0u,
            "Price a multi-leg contract. legs = [(type, strike, qty), ...] with type in call|put|stock.\n"
            "backend: auto (cuda if available, else cpu_mt) | cuda | cpu (1 thread) | cpu_mt (threads, 0 = all).");

    mod.def("greeks", &greeks, py::arg("spot"), py::arg("rate"), py::arg("vol"), py::arg("maturity"),
            py::arg("legs"), py::arg("div") = 0.0, py::arg("style") = "european", py::arg("barrier") = 0.0,
            py::arg("n_paths") = 1u << 20, py::arg("n_steps") = 252, py::arg("seed") = 42,
            py::arg("backend") = "auto", py::arg("threads") = 0u,
            "Bump-and-revalue delta/gamma/vega with common random numbers.");
}
