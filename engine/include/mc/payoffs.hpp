#pragma once

#include "mc/types.hpp"

namespace mc {

MC_HD double leg_payoff(const Leg& leg, double s) {
    double v;
    switch (leg.type) {
        case OptionType::Call: v = s > leg.strike ? s - leg.strike : 0.0; break;
        case OptionType::Put: v = leg.strike > s ? leg.strike - s : 0.0; break;
        default: v = s; break;  // Stock
    }
    return leg.qty * v;
}

// Payoff of the whole contract given the path summary.
//   s_terminal - S(T)
//   s_average  - arithmetic average over monitoring dates (only used for Asian)
//   knocked    - true if the barrier was breached along the path
MC_HD double contract_payoff(const ContractSpec& spec, double s_terminal, double s_average, bool knocked) {
    if (knocked) return 0.0;
    const double s = spec.style == PayoffStyle::AsianArithmetic ? s_average : s_terminal;
    double total = 0.0;
    for (int i = 0; i < spec.n_legs; ++i) total += leg_payoff(spec.legs[i], s);
    return total;
}

MC_HD bool is_path_dependent(PayoffStyle style) { return style != PayoffStyle::European; }

MC_HD bool breaches(PayoffStyle style, double barrier, double s) {
    return (style == PayoffStyle::BarrierUpOut && s >= barrier) ||
           (style == PayoffStyle::BarrierDownOut && s <= barrier);
}

}  // namespace mc
