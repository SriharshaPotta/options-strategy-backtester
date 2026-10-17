#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mc/cpu_engine.hpp"
#include "mc/gpu_engine.cuh"
#include "mc/stats.hpp"

namespace {

const mc::MarketParams kMarket{100.0, 0.05, 0.01, 0.2, 1.0};

class GpuEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!mc::gpu::available()) GTEST_SKIP() << "no CUDA device";
    }
};

mc::ContractSpec call(double k) {
    mc::ContractSpec s;
    s.n_legs = 1;
    s.legs[0] = {mc::OptionType::Call, k, 1.0};
    return s;
}

}  // namespace

TEST_F(GpuEngineTest, EuropeanCallMatchesBlackScholes) {
    const auto r = mc::gpu::default_engine().price(kMarket, call(100.0), {1u << 20, 64, 5});
    EXPECT_NEAR(r.price, mc::black_scholes(mc::OptionType::Call, kMarket, 100.0), 3.0 * r.std_error);
}

TEST_F(GpuEngineTest, AgreesWithCpu) {
    mc::ContractSpec ic;
    ic.n_legs = 4;
    ic.legs[0] = {mc::OptionType::Put, 85.0, 1.0};
    ic.legs[1] = {mc::OptionType::Put, 92.0, -1.0};
    ic.legs[2] = {mc::OptionType::Call, 108.0, -1.0};
    ic.legs[3] = {mc::OptionType::Call, 115.0, 1.0};
    const mc::SimConfig cfg{1u << 19, 32, 21};
    const auto g = mc::gpu::default_engine().price(kMarket, ic, cfg);
    const auto c = mc::cpu::price(kMarket, ic, cfg);
    EXPECT_NEAR(g.price, c.price, 3.0 * std::hypot(g.std_error, c.std_error));
}

TEST_F(GpuEngineTest, AsianAndBarrierBelowVanilla) {
    auto asian = call(100.0);
    asian.style = mc::PayoffStyle::AsianArithmetic;
    auto ko = call(100.0);
    ko.style = mc::PayoffStyle::BarrierUpOut;
    ko.barrier = 130.0;
    const mc::SimConfig cfg{1u << 18, 252, 8};
    const double vanilla = mc::black_scholes(mc::OptionType::Call, kMarket, 100.0);
    EXPECT_LT(mc::gpu::default_engine().price(kMarket, asian, cfg).price, vanilla);
    EXPECT_LT(mc::gpu::default_engine().price(kMarket, ko, cfg).price, vanilla);
}

TEST_F(GpuEngineTest, ReproducibleAcrossCalls) {
    const mc::SimConfig cfg{1u << 18, 16, 77};
    const auto a = mc::gpu::default_engine().price(kMarket, call(100.0), cfg);
    const auto b = mc::gpu::default_engine().price(kMarket, call(100.0), cfg);
    EXPECT_DOUBLE_EQ(a.price, b.price);
}

TEST_F(GpuEngineTest, TerminalOutputHasExpectedMean) {
    std::vector<float> terminal;
    mc::gpu::default_engine().price(kMarket, call(100.0), {1u << 18, 8, 9}, &terminal);
    ASSERT_EQ(terminal.size(), 1u << 18);
    double mean = 0.0;
    for (float s : terminal) mean += s;
    mean /= terminal.size();
    EXPECT_NEAR(mean, kMarket.spot * std::exp((kMarket.rate - kMarket.div) * kMarket.maturity), 0.3);
}
