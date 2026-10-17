"""Event-driven options strategy backtest over reconstructed chains.

Every `entry_every` trading days a chain is rebuilt from the underlying's price
and realized vol, the best-ranked setup for the chosen strategy is opened at the
natural price, and the Monte Carlo engine prices the position's expiry payoff to
record its model value and probability of profit. Positions are marked daily and
closed at a profit target, a stop loss, or settled at expiry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import engine
from .chains import SmileModel, SpreadModel, expiry_for_dte, realized_vol, synthetic_chain
from .greeks import bs_greeks, bs_price
from .metrics import summarize
from .ranker import CONTRACT_MULTIPLIER, RankWeights, rank_setups
from .strategies import get_strategy


@dataclass
class BacktestConfig:
    ticker: str = "SPY"
    strategy: str = "iron_condor"
    start: str = "2022-01-01"
    end: str | None = None
    dte: int | None = None           # default: the template's DTE
    entry_every: int = 5             # trading days between entries
    max_open: int = 4
    profit_target: float = 0.5       # close at +50% of the premium basis
    stop_loss: float = 2.0           # close at -200% of the premium basis
    contracts: int = 1
    commission: float = 0.65         # per contract per leg per side
    rate: float = 0.04
    vol_window: int = 30
    vol_premium: float = 1.15        # implied / realized multiplier
    min_iv: float = 0.08
    mc_paths: int = 200_000
    mc_backend: str = "auto"
    initial_capital: float = 100_000.0
    weights: RankWeights = field(default_factory=RankWeights)


@dataclass
class _Position:
    entry_date: date
    expiry: date
    legs: list[tuple[str, float, float]]  # (kind, strike, signed qty)
    legs_desc: str
    entry_cash: float  # per share, >0 credit
    basis: float       # per share, |premium|
    entry_spot: float
    iv: float
    model_value: float
    mc_pop: float
    mc_std_error: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict[str, float]
    engine_backend: str

    def to_dict(self) -> dict:
        trades = self.trades.copy()
        for col in ("entry_date", "exit_date", "expiry"):
            if col in trades:
                trades[col] = trades[col].astype(str)
        return {
            "config": {k: v for k, v in asdict(self.config).items() if k != "weights"},
            "metrics": self.metrics,
            "engine_backend": self.engine_backend,
            "equity": [{"date": d.strftime("%Y-%m-%d"), "equity": float(v)} for d, v in self.equity.items()],
            "trades": trades.replace({np.nan: None}).to_dict("records"),
        }


def _close_cash(pos: _Position, spot: float, asof: date, atm_iv: float, rate: float, smile: SmileModel,
                spreads: SpreadModel) -> tuple[float, float]:
    """(cash to close at natural prices, mid value) per share."""
    t = (pos.expiry - asof).days / 365.0
    if t <= 0:
        value = float(engine.payoff_at_expiry(pos.legs, np.array([spot]))[0])
        return value, value
    cash = mid_total = 0.0
    for kind, strike, qty in pos.legs:
        iv = float(smile.iv(atm_iv, spot, np.array([strike]), t)[0])
        mid = float(bs_price(spot, strike, t, rate, iv, 0.0, kind))
        delta = float(bs_greeks(spot, strike, t, rate, iv, 0.0, kind)["delta"])
        half = float(spreads.half_spread(np.array([mid]), np.array([delta]), (pos.expiry - asof).days)[0])
        cash += qty * mid - abs(qty) * half  # sell longs at bid, buy back shorts at ask
        mid_total += qty * mid
    return cash, mid_total


def run_backtest(close: pd.Series, cfg: BacktestConfig, smile: SmileModel = SmileModel(),
                 spreads: SpreadModel = SpreadModel()) -> BacktestResult:
    tpl = get_strategy(cfg.strategy)
    dte = cfg.dte or tpl.dte
    close = close.dropna().astype(float)
    iv_series = (realized_vol(close, cfg.vol_window) * cfg.vol_premium).clip(lower=cfg.min_iv)
    dates = close.index[cfg.vol_window:]
    if len(dates) < 2:
        raise ValueError("not enough price history for the volatility window")

    scale = CONTRACT_MULTIPLIER * cfg.contracts
    open_pos: list[_Position] = []
    trades: list[dict] = []
    equity: dict[pd.Timestamp, float] = {}
    realized = 0.0
    backend = "none"

    def record_close(pos: _Position, exit_day: date, exit_spot: float, exit_cash: float, reason: str):
        nonlocal realized
        n_legs = len(pos.legs)
        fees = cfg.commission * n_legs * cfg.contracts * (1 if reason == "expiry" else 2)
        pnl = (pos.entry_cash + exit_cash) * scale - fees
        realized += pnl
        trades.append({
            "entry_date": pos.entry_date, "exit_date": exit_day, "expiry": pos.expiry, "strategy": tpl.key,
            "legs": pos.legs_desc, "entry_spot": pos.entry_spot, "exit_spot": exit_spot, "iv": pos.iv,
            "entry_cash": pos.entry_cash * scale, "exit_cash": exit_cash * scale, "fees": fees, "pnl": pnl,
            "reason": reason, "days_held": (exit_day - pos.entry_date).days,
            "model_value": pos.model_value * scale, "edge": (pos.entry_cash + pos.model_value) * scale,
            "mc_pop": pos.mc_pop, "mc_std_error": pos.mc_std_error * scale,
        })

    prev_spot = float(close.loc[dates[0]])
    for i, ts in enumerate(dates):
        today = ts.date()
        spot = float(close.loc[ts])
        iv = float(iv_series.loc[ts])

        # ---- manage open positions
        still_open = []
        for pos in open_pos:
            if today >= pos.expiry:
                settle_spot = spot if today == pos.expiry else prev_spot
                value = float(engine.payoff_at_expiry(pos.legs, np.array([settle_spot]))[0])
                record_close(pos, pos.expiry, settle_spot, value, "expiry")
                continue
            cash, _ = _close_cash(pos, spot, today, iv, cfg.rate, smile, spreads)
            pnl_share = pos.entry_cash + cash
            if pnl_share >= cfg.profit_target * pos.basis:
                record_close(pos, today, spot, cash, "target")
            elif pnl_share <= -cfg.stop_loss * pos.basis:
                record_close(pos, today, spot, cash, "stop")
            else:
                still_open.append(pos)
        open_pos = still_open

        # ---- open a new position
        if i % cfg.entry_every == 0 and len(open_pos) < cfg.max_open and i < len(dates) - 1:
            chain = synthetic_chain(cfg.ticker, spot, today, expiry_for_dte(today, dte), iv, cfg.rate,
                                    smile=smile, spreads=spreads)
            ranked = rank_setups(chain, [tpl.key], cfg.weights, top_n=1)
            if not ranked.empty:
                best = ranked.iloc[0]
                legs = [(leg["kind"], leg["strike"], 1.0 if leg["side"] == "long" else -1.0) for leg in best["legs"]]
                entry_cash = best["entry_cash"] / CONTRACT_MULTIPLIER
                mc = engine.price(spot, cfg.rate, iv, chain.t, legs, n_paths=cfg.mc_paths, n_steps=1,
                                  seed=1000 + i, backend=cfg.mc_backend, return_terminal=True)
                backend = mc["backend"]
                pnl_paths = engine.payoff_at_expiry(legs, mc["terminal"]) + entry_cash
                open_pos.append(_Position(
                    entry_date=today, expiry=chain.expiry, legs=legs, legs_desc=best["legs_desc"],
                    entry_cash=entry_cash, basis=max(abs(entry_cash), 0.01), entry_spot=spot, iv=iv,
                    model_value=mc["price"],  # MC fair value of the legs we hold (negative when short)
                    mc_pop=float((pnl_paths > 0).mean()), mc_std_error=mc["std_error"],
                ))

        # ---- mark to market
        unrealized = 0.0
        for pos in open_pos:
            _, mid = _close_cash(pos, spot, today, iv, cfg.rate, smile, spreads)
            unrealized += (pos.entry_cash + mid) * scale
        equity[ts] = realized + unrealized
        prev_spot = spot

    last_ts = dates[-1]
    for pos in open_pos:
        cash, _ = _close_cash(pos, prev_spot, last_ts.date(), float(iv_series.loc[last_ts]), cfg.rate, smile, spreads)
        record_close(pos, last_ts.date(), prev_spot, cash, "end")
    equity[last_ts] = realized

    trades_df = pd.DataFrame(trades)
    equity_s = pd.Series(equity).sort_index()
    return BacktestResult(cfg, trades_df, equity_s, summarize(trades_df, equity_s, cfg.initial_capital), backend)
