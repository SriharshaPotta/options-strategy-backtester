#include "mc/gpu_engine.cuh"

#include <algorithm>

#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "mc/payoffs.hpp"
#include "mc/stats.hpp"

namespace mc::gpu {

using detail::kBlockSize;
using detail::RngState;

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string device_name() {
    if (!available()) return {};
    int device = 0;
    cudaDeviceProp props{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&props, device) != cudaSuccess) return {};
    return props.name;
}

struct Engine::Impl {
    int device = 0;
    int max_threads = 0;  // enough threads to fill every SM once
    cudaStream_t stream = nullptr;
    cudaEvent_t ev_start = nullptr, ev_rng = nullptr, ev_stop = nullptr;
    DeviceBuffer<RngState> states;
    DeviceBuffer<double> accum;
    DeviceBuffer<float> terminal;
};

Engine::Engine(int device) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const int per_sm = (props.maxThreadsPerMultiProcessor / kBlockSize) * kBlockSize;
    impl_->max_threads = props.multiProcessorCount * per_sm;

    // One RNG state per resident thread. Allocated once and reused across runs.
    impl_->states.reserve(static_cast<std::size_t>(impl_->max_threads));
    impl_->accum.reserve(2);

    CUDA_CHECK(cudaStreamCreate(&impl_->stream));
    CUDA_CHECK(cudaEventCreate(&impl_->ev_start));
    CUDA_CHECK(cudaEventCreate(&impl_->ev_rng));
    CUDA_CHECK(cudaEventCreate(&impl_->ev_stop));
}

Engine::~Engine() {
    if (!impl_) return;
    cudaEventDestroy(impl_->ev_start);
    cudaEventDestroy(impl_->ev_rng);
    cudaEventDestroy(impl_->ev_stop);
    cudaStreamDestroy(impl_->stream);
}

int Engine::threads() const { return impl_->max_threads; }

PriceResult Engine::price(const MarketParams& market, const ContractSpec& spec, const SimConfig& cfg,
                          std::vector<float>* terminal_out) {
    Impl& s = *impl_;
    CUDA_CHECK(cudaSetDevice(s.device));

    const unsigned long long wanted = (cfg.n_paths + kBlockSize - 1) / kBlockSize * kBlockSize;
    const int n_threads = static_cast<int>(std::min<unsigned long long>(wanted, s.max_threads));
    const int grid = n_threads / kBlockSize;

    float* d_terminal = nullptr;
    if (terminal_out) {
        s.terminal.reserve(cfg.n_paths);
        d_terminal = s.terminal.get();
    }

    const GbmStep gbm = make_gbm_step(market, spec.barrier, cfg.n_steps);
    detail::upload_constants(spec, gbm, s.stream);

    CUDA_CHECK(cudaEventRecord(s.ev_start, s.stream));
    detail::launch_init_rng(s.states.get(), cfg.seed, n_threads, s.stream);
    CUDA_CHECK(cudaEventRecord(s.ev_rng, s.stream));

    CUDA_CHECK(cudaMemsetAsync(s.accum.get(), 0, 2 * sizeof(double), s.stream));
    detail::launch_simulate(is_path_dependent(spec.style), grid, s.states.get(), cfg.n_paths, cfg.n_steps,
                            s.accum.get(), d_terminal, s.stream);

    double h_accum[2];
    CUDA_CHECK(cudaMemcpyAsync(h_accum, s.accum.get(), sizeof(h_accum), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaEventRecord(s.ev_stop, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    PriceResult r;
    finalize(r, h_accum[0], h_accum[1], cfg.n_paths, gbm.discount);
    r.n_steps = cfg.n_steps;
    float rng_ms = 0.f, sim_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&rng_ms, s.ev_start, s.ev_rng));
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, s.ev_rng, s.ev_stop));
    r.rng_init_ms = rng_ms;
    r.elapsed_ms = sim_ms;

    if (terminal_out) {
        terminal_out->resize(cfg.n_paths);
        CUDA_CHECK(cudaMemcpy(terminal_out->data(), d_terminal, cfg.n_paths * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return r;
}

Engine& default_engine() {
    static Engine engine(0);
    return engine;
}

}  // namespace mc::gpu
