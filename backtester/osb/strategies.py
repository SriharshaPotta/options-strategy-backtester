"""Strategy templates, expressed as delta targets so they adapt to any underlying."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegTemplate:
    kind: str           # "call" | "put"
    side: int           # +1 long, -1 short
    target_delta: float  # unsigned, e.g. 0.30


@dataclass(frozen=True)
class StrategyTemplate:
    key: str
    label: str
    legs: tuple[LegTemplate, ...]
    dte: int = 30
    # Strikes must strictly increase in leg order (spreads/condors). Straddles opt out.
    ordered_strikes: bool = True
    same_strike: bool = False
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


P, C = "put", "call"

STRATEGIES: dict[str, StrategyTemplate] = {
    s.key: s
    for s in [
        StrategyTemplate("short_put", "Short Put", (LegTemplate(P, -1, 0.30),),
                         description="Sell a 30-delta put; collect premium, long the underlying below strike.",
                         tags=("credit", "bullish")),
        StrategyTemplate("bull_put_spread", "Bull Put Spread", (LegTemplate(P, +1, 0.15), LegTemplate(P, -1, 0.30)),
                         description="Sell 30-delta put, buy 15-delta put for defined risk.",
                         tags=("credit", "bullish", "defined-risk")),
        StrategyTemplate("bear_call_spread", "Bear Call Spread",
                         (LegTemplate(C, -1, 0.30), LegTemplate(C, +1, 0.15)),
                         description="Sell 30-delta call, buy 15-delta call.",
                         tags=("credit", "bearish", "defined-risk")),
        StrategyTemplate("iron_condor", "Iron Condor",
                         (LegTemplate(P, +1, 0.10), LegTemplate(P, -1, 0.16),
                          LegTemplate(C, -1, 0.16), LegTemplate(C, +1, 0.10)),
                         description="Short 16-delta strangle with 10-delta wings.",
                         tags=("credit", "neutral", "defined-risk")),
        StrategyTemplate("short_strangle", "Short Strangle", (LegTemplate(P, -1, 0.16), LegTemplate(C, -1, 0.16)),
                         description="Sell 16-delta put and call.", tags=("credit", "neutral")),
        StrategyTemplate("bull_call_spread", "Bull Call Spread",
                         (LegTemplate(C, +1, 0.50), LegTemplate(C, -1, 0.30)),
                         description="Buy ATM call, sell 30-delta call.", tags=("debit", "bullish", "defined-risk")),
        StrategyTemplate("long_straddle", "Long Straddle", (LegTemplate(P, +1, 0.50), LegTemplate(C, +1, 0.50)),
                         ordered_strikes=False, same_strike=True,
                         description="Buy ATM put and call; long volatility.", tags=("debit", "long-vol")),
    ]
}


def get_strategy(key: str) -> StrategyTemplate:
    try:
        return STRATEGIES[key]
    except KeyError:
        raise KeyError(f"unknown strategy '{key}'; choose from {sorted(STRATEGIES)}") from None
