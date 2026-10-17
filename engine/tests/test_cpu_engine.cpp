#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mc/cpu_engine.hpp"
#include "mc/greeks.hpp"
#include "mc/payoffs.hpp"
#include "mc/stats.hpp"

namespace {

const mc::MarketParams kMarket{100.0, 0.05, 0.01, 0.2, 1.0};
const mc::SimConfig kCfg{200'000, 50, 7};

mc::ContractSpec single(mc::OptionType t, double k, double qty = 1.0) {
    mc::ContractSpec s;
    s.n_legs = 1;
    s.legs[0] = {t, k, qty};
    return s;
}

void expect_within_se(const mc::PriceResult& r, double ref, double n_se = 3.0) {
    EXPECT_NEAR(r.price, ref, n_se * r.std_error) << "MC " << r.price << " +/- " << r.std_error << " vs " << ref;
}

}  // namespace

TEST(Payoffs, LegAndContract) {
    mc::ContractSpec s;
    s.n_legs = 2;
    s.legs[0] = {mc::OptionType::Put, 95.0, -1.0};
    s.legs[1] = {mc::OptionType::Put, 90.0, 1.0};
    EXPECT_DOUBLE_EQ(mc::contract_payoff(s, 100.0, 0.0, false), 0.0);
    EXPECT_DOUBLE_EQ(mc::contract_payoff(s, 93.0, 0.0, false), -2.0);
    EXPECT_DOUBLE_EQ(mc::contract_payoff(s, 80.0, 0.0, false), -5.0);
    EXPECT_DOUBLE_EQ(mc::contract_payoff(s, 80.0, 0.0, true), 0.0);
}

TEST(BlackScholes, KnownValues) {
    const mc::MarketParams m{100.0, 0.05, 0.0, 0.2, 1.0};
    EXPECT_NEAR(mc::black_scholes(mc::OptionType::Call, m, 100.0), 10.4506, 1e-4);
    EXPECT_NEAR(mc::black_scholes(mc::OptionType::Put, m, 100.0), 5.5735, 1e-4);
}

TEST(CpuEngine, EuropeanCallMatchesBlackScholes) {
    for (double k : {80.0, 100.0, 120.0}) {
        const auto r = mc::cpu::price(kMarket, single(mc::OptionType::Call, k), kCfg);
        expect_within_se(r, mc::black_scholes(mc::OptionType::Call, kMarket, k));
    }
}

TEST(CpuEngine, EuropeanPutMatchesBlackScholes) {
    for (double k : {80.0, 100.0, 120.0}) {
        const auto r = mc::cpu::price(kMarket, single(mc::OptionType::Put, k), kCfg);
        expect_within_se(r, mc::black_scholes(mc::OptionType::Put, kMarket, k));
    }
}

TEST(CpuEngine, PutCallParityWithCommonRandomNumbers) {
    // Long call + short put + short forward-stock replicate zero; with shared
    // seeds the MC estimate of C - P must equal S e^{-qT} - K e^{-rT} closely.
    const double k = 105.0;
    const auto c = mc::cpu::price(kMarket, single(mc::OptionType::Call, k), kCfg);
    const auto p = mc::cpu::price(kMarket, single(mc::OptionType::Put, k), kCfg);
    const double forward = kMarket.spot * std::exp(-kMarket.div * kMarket.maturity) -
                           k * std::exp(-kMarket.rate * kMarket.maturity);
    EXPECT_NEAR(c.price - p.price, forward, 3.0 * std::hypot(c.std_error, p.std_error));
}

TEST(CpuEngine, VerticalSpreadBoundedByWidth) {
    mc::ContractSpec s;
    s.n_legs = 2;
    s.legs[0] = {mc::OptionType::Call, 100.0, 1.0};
    s.legs[1] = {mc::OptionType::Call, 110.0, -1.0};
    const auto r = mc::cpu::price(kMarket, s, kCfg);
    EXPECT_GT(r.price, 0.0);
    EXPECT_LT(r.price, 10.0 * std::exp(-kMarket.rate * kMarket.maturity));
    expect_within_se(r, mc::black_scholes(s, kMarket));
}

TEST(CpuEngine, IronCondorMatchesBlackScholes) {
    mc::ContractSpec s;
    s.n_legs = 4;
    s.legs[0] = {mc::OptionType::Put, 85.0, 1.0};
    s.legs[1] = {mc::OptionType::Put, 92.0, -1.0};
    s.legs[2] = {mc::OptionType::Call, 108.0, -1.0};
    s.legs[3] = {mc::OptionType::Call, 115.0, 1.0};
    expect_within_se(mc::cpu::price(kMarket, s, kCfg), mc::black_scholes(s, kMarket));
}

TEST(CpuEngine, AsianCheaperThanEuropean) {
    auto asian = single(mc::OptionType::Call, 100.0);
    asian.style = mc::PayoffStyle::AsianArithmetic;
    const auto a = mc::cpu::price(kMarket, asian, kCfg);
    EXPECT_LT(a.price, mc::black_scholes(mc::OptionType::Call, kMarket, 100.0));
    EXPECT_GT(a.price, 0.0);
}

TEST(CpuEngine, KnockOutCheaperThanVanilla) {
    auto barrier = single(mc::OptionType::Call, 100.0);
    barrier.style = mc::PayoffStyle::BarrierUpOut;
    barrier.barrier = 130.0;
    const auto b = mc::cpu::price(kMarket, barrier, kCfg);
    EXPECT_LT(b.price, mc::black_scholes(mc::OptionType::Call, kMarket, 100.0));
    EXPECT_GT(b.price, 0.0);
}

TEST(CpuEngine, DeterministicForSeed) {
    const auto s = single(mc::OptionType::Call, 100.0);
    const auto a = mc::cpu::price(kMarket, s, {10'000, 10, 99});
    const auto b = mc::cpu::price(kMarket, s, {10'000, 10, 99});
    EXPECT_DOUBLE_EQ(a.price, b.price);
}

TEST(CpuEngine, TerminalOutput) {
    std::vector<float> terminal;
    mc::cpu::price(kMarket, single(mc::OptionType::Call, 100.0), {50'000, 10, 3}, &terminal);
    ASSERT_EQ(terminal.size(), 50'000u);
    double mean = 0.0;
    for (float s : terminal) mean += s;
    mean /= terminal.size();
    // E[S_T] = S0 exp((r - q) T)
    EXPECT_NEAR(mean, kMarket.spot * std::exp((kMarket.rate - kMarket.div) * kMarket.maturity), 0.5);
}

TEST(Greeks, BumpDeltaCloseToAnalytic) {
    const auto spec = single(mc::OptionType::Call, 100.0);
    const auto g = mc::bump_greeks([](auto&&... a) { return mc::cpu::price(a...); }, kMarket, spec,
                                   mc::SimConfig{200'000, 1, 11});
    const double st = kMarket.vol * std::sqrt(kMarket.maturity);
    const double d1 = (std::log(kMarket.spot / 100.0) +
                       (kMarket.rate - kMarket.div + 0.5 * kMarket.vol * kMarket.vol) * kMarket.maturity) / st;
    const double delta = std::exp(-kMarket.div * kMarket.maturity) * mc::norm_cdf(d1);
    EXPECT_NEAR(g.delta, delta, 0.01);
    EXPECT_GT(g.gamma, 0.0);
    EXPECT_GT(g.vega, 0.0);
}
