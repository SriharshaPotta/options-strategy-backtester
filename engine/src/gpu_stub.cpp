// Compiled instead of the .cu sources when CMake finds no CUDA compiler, so the
// rest of the project (bindings, tests, bench) links unchanged.
#include <stdexcept>

#include "mc/gpu_engine.cuh"

namespace mc::gpu {

bool available() { return false; }
std::string device_name() { return {}; }

struct Engine::Impl {};

Engine::Engine(int) { throw std::runtime_error("mcengine was built without CUDA support"); }
Engine::~Engine() = default;
int Engine::threads() const { return 0; }

PriceResult Engine::price(const MarketParams&, const ContractSpec&, const SimConfig&, std::vector<float>*) {
    throw std::runtime_error("mcengine was built without CUDA support");
}

Engine& default_engine() { throw std::runtime_error("mcengine was built without CUDA support"); }

}  // namespace mc::gpu
