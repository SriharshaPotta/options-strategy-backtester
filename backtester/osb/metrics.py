"""Performance statistics for a backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, equity: pd.Series, initial_capital: float) -> dict[str, float]:
    pnl = trades["pnl"] if not trades.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    account = initial_capital + equity
    daily_ret = account.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0
    drawdown = account - account.cummax()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9) if len(equity) > 1 else 1.0
    total = float(pnl.sum())

    return {
        "n_trades": int(len(pnl)),
        "total_pnl": total,
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "max_drawdown_pct": float((drawdown / account.cummax()).min()) if len(drawdown) else 0.0,
        "cagr": float((1 + total / initial_capital) ** (1 / years) - 1) if total > -initial_capital else -1.0,
        "sharpe": sharpe,
        "avg_days_held": float(trades["days_held"].mean()) if not trades.empty else 0.0,
    }
