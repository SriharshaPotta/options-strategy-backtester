"""Rank concrete strategy setups by open interest, bid/ask spread and delta fit.

For every template leg we take the `k` strikes whose |delta| is closest to the
target, form every valid combination, and score each combination:

  oi_score     = log1p(min leg OI) / log1p(best min-OI in the candidate set)
  spread_score = 1 - clip(half-spread cost / |net premium| / spread_cap)
  delta_score  = 1 - clip(mean | |delta| - target | / delta_tol)
  score        = 100 * weighted mean of the available scores

Chains without open interest (reconstructed history) drop the OI term and
re-normalise the remaining weights.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .chains import Chain
from .engine import payoff_at_expiry
from .strategies import STRATEGIES, StrategyTemplate, get_strategy

CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class RankWeights:
    open_interest: float = 0.35
    spread: float = 0.35
    delta: float = 0.30
    spread_cap: float = 0.25  # half-spread cost >= 25% of premium scores 0
    delta_tol: float = 0.10   # mean delta miss >= 0.10 scores 0


def _candidates(chain: Chain, kind: str, target: float, k: int) -> pd.DataFrame:
    side = chain.side(kind)
    if side.empty:
        return side
    miss = (side["delta"].abs() - target).abs()
    return side.assign(_miss=miss).nsmallest(k, "_miss")


def _combinations(chain: Chain, tpl: StrategyTemplate, k: int):
    pools = [_candidates(chain, leg.kind, leg.target_delta, k).to_dict("records") for leg in tpl.legs]
    for combo in itertools.product(*pools):
        strikes = [row["strike"] for row in combo]
        if tpl.same_strike and len(set(strikes)) != 1:
            continue
        if tpl.ordered_strikes and any(b <= a for a, b in zip(strikes, strikes[1:])):
            continue
        yield combo


def _pop_and_extremes(chain: Chain, legs: list[tuple[str, float, float]], entry_cash: float, atm_iv: float):
    """Risk-neutral probability of profit at expiry plus max profit / max loss (per share)."""
    t = max(chain.t, 1 / 365)
    sd = atm_iv * np.sqrt(t)
    mu = np.log(chain.spot) + (chain.rate - 0.5 * atm_iv ** 2) * t
    grid = np.exp(np.linspace(mu - 6 * sd, mu + 6 * sd, 4001))
    pnl = payoff_at_expiry(legs, grid) + entry_cash
    dens = np.exp(-0.5 * ((np.log(grid) - mu) / sd) ** 2) / grid
    w = dens * np.gradient(grid)
    pop = float(w[pnl > 0].sum() / w.sum())

    strikes = sorted({k for _, k, _ in legs})
    edges = np.array([0.0, *strikes, strikes[-1] * 3.0])
    edge_pnl = payoff_at_expiry(legs, edges) + entry_cash
    net_calls = sum(q for kind, _, q in legs if kind == "call")
    max_profit = np.inf if net_calls > 0 else float(edge_pnl.max())
    max_loss = -np.inf if net_calls < 0 else float(edge_pnl.min())
    return pop, max_profit, max_loss


def evaluate(chain: Chain, tpl: StrategyTemplate, combo) -> dict:
    legs = [(row["kind"], float(row["strike"]), float(leg.side)) for row, leg in zip(combo, tpl.legs)]
    net_mid = sum(leg.side * row["mid"] for row, leg in zip(combo, tpl.legs))  # >0 debit
    # Natural fill: buy at the ask, sell at the bid. entry_cash > 0 means a credit.
    entry_cash = sum(-leg.side * (row["ask"] if leg.side > 0 else row["bid"]) for row, leg in zip(combo, tpl.legs))
    half_spread_cost = sum((row["ask"] - row["bid"]) / 2.0 for row in combo)
    atm_iv = float(np.median([row["iv"] for row in combo]))
    pop, max_profit, max_loss = _pop_and_extremes(chain, legs, entry_cash, atm_iv)
    oi = [row["open_interest"] for row in combo]

    return {
        "strategy": tpl.key,
        "label": tpl.label,
        "expiry": chain.expiry.isoformat(),
        "dte": chain.dte,
        "legs": [
            {"kind": row["kind"], "side": "long" if leg.side > 0 else "short", "strike": float(row["strike"]),
             "bid": float(row["bid"]), "ask": float(row["ask"]), "mid": float(row["mid"]),
             "delta": float(row["delta"]), "iv": float(row["iv"]),
             "open_interest": None if pd.isna(row["open_interest"]) else int(row["open_interest"])}
            for row, leg in zip(combo, tpl.legs)
        ],
        "legs_desc": " / ".join(f"{'+' if leg.side > 0 else '-'}{row['strike']:g}{row['kind'][0].upper()}"
                                for row, leg in zip(combo, tpl.legs)),
        "net_premium": float(-net_mid * CONTRACT_MULTIPLIER),  # >0 credit received at mid
        "entry_cash": float(entry_cash * CONTRACT_MULTIPLIER),
        "max_profit": float(max_profit * CONTRACT_MULTIPLIER),
        "max_loss": float(max_loss * CONTRACT_MULTIPLIER),
        "pop": pop,
        "min_oi": float(np.nan if any(pd.isna(oi)) else min(oi)),
        "spread_pct": float(half_spread_cost / max(abs(net_mid), 0.05)),
        "delta_err": float(np.mean([abs(abs(row["delta"]) - leg.target_delta) for row, leg in zip(combo, tpl.legs)])),
        "net_delta": float(sum(leg.side * row["delta"] for row, leg in zip(combo, tpl.legs))),
    }


def score(df: pd.DataFrame, weights: RankWeights = RankWeights()) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    has_oi = df["min_oi"].notna() & np.isfinite(df["min_oi"])
    best = np.log1p(df.loc[has_oi, "min_oi"].max()) if has_oi.any() else 0.0
    df["oi_score"] = np.where(has_oi, np.log1p(df["min_oi"].fillna(0)) / best if best > 0 else 0.0, np.nan)
    df["spread_score"] = 1.0 - np.clip(df["spread_pct"] / weights.spread_cap, 0.0, 1.0)
    df["delta_score"] = 1.0 - np.clip(df["delta_err"] / weights.delta_tol, 0.0, 1.0)

    w_oi = np.where(has_oi, weights.open_interest, 0.0)
    num = w_oi * df["oi_score"].fillna(0) + weights.spread * df["spread_score"] + weights.delta * df["delta_score"]
    df["score"] = 100.0 * num / (w_oi + weights.spread + weights.delta)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def rank_setups(chain: Chain, strategies: list[str] | None = None, weights: RankWeights = RankWeights(),
                k: int = 3, top_n: int | None = 25) -> pd.DataFrame:
    keys = strategies or list(STRATEGIES)
    rows = [evaluate(chain, tpl, combo) for tpl in map(get_strategy, keys) for combo in _combinations(chain, tpl, k)]
    ranked = score(pd.DataFrame(rows), weights)
    return ranked.head(top_n) if top_n else ranked
